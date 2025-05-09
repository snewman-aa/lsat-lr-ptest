def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    response_data = r.json()
    assert response_data["status"] == "ok"
    assert "database_status" in response_data


def test_list_questions_default(client):
    r = client.get("/questions")
    assert r.status_code == 200
    data = r.json()
    assert "questions" in data
    if data["questions"]:
        assert any(q["question_number"] == 1 for q in data["questions"])


def test_generate_and_retrieve_flow(client):
    user_prompt = "generate a test about cats"
    r1 = client.post("/generate", json={"prompt": user_prompt})
    assert r1.status_code == 200
    gen = r1.json()
    test_id = gen["test_id"]
    assert isinstance(test_id, int)

    sq = gen["sample_question"]
    assert sq["stimulus"]    == f"Mocked stimulus for request: '{user_prompt}'"
    assert sq["prompt"]      == "This is a mocked question prompt."
    assert sq["explanation"] == "This is a mocked explanation for the sample question."

    sims = gen["similar_questions"]
    assert sims
    assert len(sims) > 0
    assert sims[0]["question_number"] == 1001

    first_similar_question_id = sims[0]["question_number"]
    # correct answer for question 1001 in test DB is 'B'
    correct_answer_for_mocked_question = "B"

    r2 = client.post(
        "/save_test",
        json={"test_id": test_id,
              "answers": [{"question_number": first_similar_question_id, "selected_answer": correct_answer_for_mocked_question}]}
    )
    assert r2.status_code == 200
    save_response_data = r2.json()
    assert save_response_data["status"] == "ok"
    assert "message" in save_response_data
    assert f"Answers for test_id {test_id} saved successfully" in save_response_data["message"]

    r3 = client.get(f"/tests/{test_id}")
    assert r3.status_code == 200
    detail = r3.json()
    assert detail["test_id"] == test_id
    assert detail["questions"][0]["selected_answer"] == correct_answer_for_mocked_question
    assert detail["questions"][0]["question_number"] == first_similar_question_id
    assert detail["score"] == "1/1"

def test_list_tests_and_clear(client):
    r = client.get("/tests")
    assert r.status_code == 200
    lst = r.json()
    assert any(t["prompt"] == "generate a test about cats" for t in lst)

    r2 = client.delete("/tests")
    assert r2.status_code == 200
    delete_response_data = r2.json()
    assert delete_response_data["status"] == "ok"
    assert "message" in delete_response_data
    assert "All test data cleared and tables reset successfully" in delete_response_data["message"]


    r3 = client.get("/tests")
    assert r3.status_code == 200
    assert r3.json() == []