import "./style.css";

const app = document.querySelector("#app");
app.innerHTML = `
  <section class="shell">
    <header>
      <p class="eyebrow">Bone Disease CVQA</p>
      <h1>Ask about an X-ray</h1>
      <p class="muted">Web frontend → REST API → Python online pipeline</p>
    </header>
    <form id="form" class="card">
      <label> X-ray image
        <input id="image" type="file" accept=".jpg,.jpeg,.png,.dcm,.dicom,image/*" required />
      </label>
      <label> Question
        <textarea id="question" rows="3" required placeholder="Is there a fracture?"></textarea>
      </label>
      <label> Conversation history (optional JSON)
        <textarea id="history" rows="3" placeholder='[{"question":"...","answer":"..."}]'>[]</textarea>
      </label>
      <button id="submit" type="submit">Run inference</button>
      <p id="status" class="status"></p>
    </form>
    <section class="card result">
      <h2>Result</h2>
      <pre id="result">No inference yet.</pre>
    </section>
  </section>
`;

const form = document.querySelector("#form");
const status = document.querySelector("#status");
const result = document.querySelector("#result");
const button = document.querySelector("#submit");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const image = document.querySelector("#image").files[0];
  const question = document.querySelector("#question").value.trim();
  const history = document.querySelector("#history").value.trim() || "[]";
  if (!image || !question) return;

  const body = new FormData();
  body.append("image", image);
  body.append("question", question);
  body.append("history", history);
  body.append("sample_id", image.name);
  button.disabled = true;
  status.textContent = "Running model...";
  result.textContent = "";
  try {
    const response = await fetch(`${import.meta.env.VITE_API_URL || "http://localhost:8000"}/api/infer`, {
      method: "POST",
      body,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Inference failed");
    result.textContent = JSON.stringify(payload, null, 2);
    status.textContent = "Completed.";
  } catch (error) {
    status.textContent = error.message;
    result.textContent = "Request failed.";
  } finally {
    button.disabled = false;
  }
});
