from api.models.schema import AdviceRequest, AdviceResponse


def generate_advice(request: AdviceRequest) -> AdviceResponse:
    return AdviceResponse(advice="This is a test advice")
