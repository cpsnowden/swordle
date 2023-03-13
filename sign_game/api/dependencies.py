from fastapi import Request


def resolve_model(request: Request):
    return request.app.state.model
