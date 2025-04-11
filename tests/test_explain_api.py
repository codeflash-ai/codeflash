from codeflash.api.aiservice import AiServiceClient
def test_explain_api():
    aiservice = AiServiceClient()
    source_code: str = """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
"""
    dependency_code: str = "def helper(): return 1"
    trace_id: str = "d5822364-7617-4389-a4fc-64602a00b714"
    existing_explanation: str = "I used to numpy to optimize it"
    optimized_code: str = """def bubble_sort(arr):
    return arr.sort()
"""
    new_explanation = aiservice.get_new_explanation(source_code=source_code, optimized_code=optimized_code,
                existing_explanation=existing_explanation, dependency_code=dependency_code,
                trace_id=trace_id)
    print("\nNew explanation: \n", new_explanation)
    assert new_explanation.__len__()>0