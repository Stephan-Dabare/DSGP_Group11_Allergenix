document.getElementById("uploadForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  const fileInput = document.getElementById("fileInput");
  if (!fileInput.files.length) {
    alert("Please select an image!");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  const response = await fetch("/upload", {
    method: "POST",
    body: formData,
  });

  const data = await response.json();
  if (data.error) {
    alert(data.error);
    return;
  }

  // Display uploaded image
  const imageContainer = document.getElementById("imageContainer");
  imageContainer.innerHTML = `<img src="${data.filepath}" alt="Uploaded Image">`;

  // Display extracted ingredients
  const ingredients = document.getElementById("ingredients");
  ingredients.innerHTML = `<h3>Extracted Ingredients:</h3><ul>${data.results.ingredients.map(
    (ingredient) => `<li>${ingredient}</li>`
  ).join("")}</ul>`;
});