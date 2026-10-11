from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import json
import logging
import re
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.concurrency import run_in_threadpool

from api.routes import monitoring_scheduler, persistence_manager, router
from auth import authenticated_user_id, verify_access_token
from config import settings, validate_runtime_settings
from persistence.artifacts import ArtifactStorageError

logger = logging.getLogger("satellite-intelligence")
request_id_context: ContextVar[str] = ContextVar("request_id", default="-")


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", request_id_context.get()),
        }
        for key in (
            "event",
            "http_method",
            "http_route",
            "status_code",
            "error_type",
            "database_error_sqlstate",
            "database_error_schema",
            "database_error_table",
            "database_error_column",
            "database_error_constraint",
            "database_error_statement",
            "schema_issues",
            "issue_count",
            "job_id",
            "analysis",
        ):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


if not logger.handlers:
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(JsonLogFormatter())
    logger.addHandler(log_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def _allowed_origin_header(request: Request) -> dict[str, str]:
    origin = request.headers.get("origin")
    if origin and origin in settings.cors_origins:
        return {"Access-Control-Allow-Origin": origin, "Vary": "Origin"}
    return {}


def _database_error_context(
    exc: SQLAlchemyError, *, include_statement: bool = False
) -> dict[str, str]:
    original = getattr(exc, "orig", None)
    sqlstate = getattr(original, "sqlstate", None) or getattr(original, "pgcode", None)
    context: dict[str, str] = {}
    if isinstance(sqlstate, str) and re.fullmatch(r"[A-Z0-9]{5}", sqlstate):
        context["database_error_sqlstate"] = sqlstate

    diagnostics = getattr(original, "diag", None)
    for diagnostic_name in ("schema_name", "table_name", "column_name", "constraint_name"):
        value = getattr(diagnostics, diagnostic_name, None)
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_$.-]{1,128}", value):
            context[f"database_error_{diagnostic_name.removesuffix('_name')}"] = value
    statement = getattr(exc, "statement", None)
    if include_statement and isinstance(statement, str):
        normalized_statement = re.sub(r"\s+", " ", statement).strip()
        if normalized_statement:
            context["database_error_statement"] = normalized_statement[:512]
    return context


def _find_database_error(exc: BaseException) -> SQLAlchemyError | None:
    pending = [exc]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        if isinstance(current, SQLAlchemyError):
            return current
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return None


startup_issues = validate_runtime_settings()
if startup_issues:
    logger.warning(
        "Runtime configuration validation failed.",
        extra={"event": "configuration.invalid", "issue_count": len(startup_issues)},
    )

@asynccontextmanager
async def lifespan(_: FastAPI):
    if persistence_manager.repository:
        try:
            persistence_manager.repository.check_connection()
        except SQLAlchemyError as exc:
            logger.error(
                "Metadata database is unavailable at startup.",
                extra={
                    "event": "database.startup_check_failed",
                    "error_type": type(exc).__name__,
                    **_database_error_context(exc),
                },
            )
        else:
            try:
                schema_available = persistence_manager.repository.check_schema()
            except SQLAlchemyError as exc:
                logger.error(
                    "Metadata database schema could not be verified at startup.",
                    extra={
                        "event": "database.schema_check_failed",
                        "error_type": type(exc).__name__,
                        **_database_error_context(exc),
                    },
                )
            else:
                if not schema_available:
                    logger.error(
                        "Metadata database is missing required application tables or columns.",
                        extra={
                            "event": "database.schema_incompatible",
                            "error_type": "DatabaseSchemaMismatch",
                        },
                    )
                else:
                    try:
                        interrupted = persistence_manager.mark_interrupted_jobs()
                    except SQLAlchemyError as exc:
                        logger.error(
                            "Interrupted analysis jobs could not be recovered at startup.",
                            extra={
                                "event": "analysis.recovery_failed",
                                "error_type": type(exc).__name__,
                                **_database_error_context(
                                    exc, include_statement=True
                                ),
                            },
                        )
                    else:
                        if interrupted:
                            logger.warning(
                                "Interrupted analysis jobs were marked failed.",
                                extra={
                                    "event": "analysis.interrupted_jobs_recovered",
                                    "issue_count": interrupted,
                                },
                            )
    if persistence_manager.artifact_store.name == "supabase":
        try:
            persistence_manager.artifact_store.check()
        except ArtifactStorageError as exc:
            logger.error(
                "Supabase Storage is unavailable at startup.",
                extra={
                    "event": "storage.startup_check_failed",
                    "error_type": type(exc).__name__,
                },
            )
    if (
        settings.satellite_monitor_enabled
        and monitoring_scheduler is not None
        and persistence_manager.repository is not None
    ):
        try:
            if persistence_manager.repository.check_schema():
                await monitoring_scheduler.start()
            else:
                logger.error(
                    "Satellite monitoring scheduler is disabled because its database migration is not applied.",
                    extra={"event": "monitor.schema_unavailable"},
                )
        except (SQLAlchemyError, ValueError) as exc:
            logger.error(
                "Satellite monitoring scheduler could not start.",
                extra={
                    "event": "monitor.startup_failed",
                    "error_type": type(exc).__name__,
                },
            )
    try:
        yield
    finally:
        if monitoring_scheduler is not None:
            await monitoring_scheduler.stop()
        persistence_manager.close()


app = FastAPI(
    title="Satellite Intelligence API",
    version="1.0.0",
    description="Stage-A satellite intelligence pipeline with safe local/live data handling.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID", "Idempotency-Key"],
    expose_headers=["X-Request-ID"],
)
app.include_router(router, prefix="/api")


async def request_logging_middleware(request: Request, call_next):
    supplied_id = request.headers.get("x-request-id", "")
    request_id = (
        supplied_id
        if re.fullmatch(r"[A-Za-z0-9._-]{1,64}", supplied_id)
        else uuid4().hex
    )
    context_token = request_id_context.set(request_id)
    try:
        response = await call_next(request)
        route = request.scope.get("route")
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "HTTP request completed.",
            extra={
                "event": "http.request",
                "http_method": request.method,
                "http_route": getattr(route, "path", "unmatched"),
                "status_code": response.status_code,
            },
        )
        return response
    except Exception as exc:
        logger.error(
            "HTTP request failed before a response was generated.",
            extra={
                "event": "http.request_failed",
                "http_method": request.method,
                "http_route": "unmatched",
                "error_type": type(exc).__name__,
            },
        )
        raise
    finally:
        request_id_context.reset(context_token)


@app.middleware("http")
async def authentication_middleware(request: Request, call_next):
    public_paths = {"/", "/api/health", "/api/ready", "/api/auth/status"}
    if (
        not settings.authentication_required
        or request.method == "OPTIONS"
        or request.url.path in public_paths
    ):
        return await call_next(request)

    authorization = request.headers.get("authorization", "")
    scheme, separator, access_token = authorization.partition(" ")
    if (
        not separator
        or scheme.lower() != "bearer"
        or not access_token
        or access_token.strip() != access_token
    ):
        response_headers = {
            **_allowed_origin_header(request),
            "WWW-Authenticate": "Bearer",
        }
        return JSONResponse(
            status_code=401,
            content={"detail": "A valid Supabase sign-in is required."},
            headers=response_headers,
        )

    try:
        user = await run_in_threadpool(verify_access_token, access_token)
    except HTTPException as exc:
        response_headers = _allowed_origin_header(request)
        if exc.headers:
            response_headers.update(exc.headers)
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=response_headers,
        )

    request.state.authenticated_user = user
    context_token = authenticated_user_id.set(user.id)
    try:
        return await call_next(request)
    finally:
        authenticated_user_id.reset(context_token)


app.middleware("http")(request_logging_middleware)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    route = request.scope.get("route")
    logger.warning(
        "Request validation failed.",
        extra={
            "event": "http.validation_failed",
            "http_route": getattr(route, "path", "unmatched"),
            "status_code": 422,
            "issue_count": len(exc.errors()),
        },
    )
    errors = [
        {
            "loc": error.get("loc", ()),
            "msg": error.get("msg", "Invalid value."),
            "type": error.get("type", "value_error"),
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"detail": "Request validation failed.", "errors": errors},
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    route = request.scope.get("route")
    database_error = _find_database_error(exc)
    logger.error(
        "Unhandled application error.",
        extra={
            "event": "http.unhandled_error",
            "http_route": getattr(route, "path", "unmatched"),
            "status_code": 500,
            "error_type": type(database_error or exc).__name__,
            **(
                _database_error_context(database_error)
                if database_error is not None
                else {}
            ),
        },
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Analysis could not be completed because of an internal processing error."},
        headers=_allowed_origin_header(request),
    )


@app.get("/")
def root() -> dict[str, str]:
    return {"name": "Satellite Intelligence API", "health": "/api/health", "ready": "/api/ready"}
