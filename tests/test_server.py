def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_list_questions_default(client):
    r = client.get("/questions")
    assert r.status_code == 200
    data = r.json()
    assert "questions" in data
    assert any(q["question_number"] == 1 for q in data["questions"])


def test_generate_and_retrieve_flow(client):
    r1 = client.post("/generate", json={"prompt": "foo"})
    assert r1.status_code == 200
    gen = r1.json()
    test_id = gen["test_id"]

    sq = gen["sample_question"]
    assert sq["stimulus"]    == "S"
    assert sq["prompt"]      == "P"
    assert sq["explanation"] == "E"

    sims = gen["similar_questions"]
    assert sims and sims[0]["question_number"] == 1

    r2 = client.post(
        "/save_test",
        json={"test_id": test_id,
              "answers": [{"question_number": 1, "selected_answer": "A"}]}
    )
    assert r2.status_code == 200
    assert r2.json() == {"status": "ok", "test_id": test_id}

    r3 = client.get(f"/tests/{test_id}")
    assert r3.status_code == 200
    detail = r3.json()
    assert detail["test_id"] == test_id
    assert detail["questions"][0]["selected_answer"] == "A"
    assert detail["score"] == "1/1"


def test_list_tests_and_clear(client):
    r = client.get("/tests")
    assert r.status_code == 200
    lst = r.json()
    assert any(t["prompt"] == "foo" for t in lst)

    r2 = client.delete("/tests")
    assert r2.status_code == 200
    assert r2.json() == {"status": "ok"}

    r3 = client.get("/tests")
    assert r3.status_code == 200
    assert r3.json() == []
