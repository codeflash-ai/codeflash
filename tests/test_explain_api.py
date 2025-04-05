from codeflash.api.aiservice import AiServiceClient
from codeflash.models.ExperimentMetadata import ExperimentMetadata
def test_explain_api():
    aiservice = AiServiceClient()
    source_code: str = "a"
    dependency_code: str = "b"
    trace_id: str = "d5822364-7617-4389-a4fc-64602a00b714"
    num_candidates: int = 1
    experiment_metadata: ExperimentMetadata | None = None
    existing_explanation: str = "some explanation"
    new_explanation = aiservice.get_new_explanation(source_code=source_code,
                dependency_code=dependency_code,
                trace_id=trace_id,
                num_candidates=num_candidates,
                experiment_metadata=experiment_metadata, existing_explanation=existing_explanation)
    assert new_explanation.__len__()>0