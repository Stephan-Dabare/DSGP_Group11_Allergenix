function toggleMenu() {
    document.querySelector('.nav-links').classList.toggle('show');
}

document.addEventListener("DOMContentLoaded", function () {
    const modal = document.getElementById("photoTipsModal");
    const openModal = document.querySelector(".tip");
    const closeModal = document.querySelector(".close");

    openModal.addEventListener("click", function () {
        modal.style.display = "flex";
        document.body.style.overflow = "hidden"; // Prevent scrolling
    });

    closeModal.addEventListener("click", function () {
        modal.style.display = "none";
        document.body.style.overflow = "auto"; // Restore scrolling
    });

    window.addEventListener("click", function (event) {
        if (event.target === modal) {
            modal.style.display = "none";
            document.body.style.overflow = "auto";
        }
    });
});

let imageCropper;
const fileUploader = document.getElementById('fileUploader');
const cropModal = document.getElementById('cropModal');
const cropTarget = document.getElementById('cropTarget');
const confirmCrop = document.getElementById('confirmCrop');
const ingredientDisplay = document.getElementById("ingredientDisplay");
const overlay = document.getElementById("overlay");
const ingredientList = document.getElementById("ingredientList");

// Open file input when clicking "Use Photo"
document.querySelector(".upload-btn").addEventListener("click", function () {
    fileUploader.click();
});

// Handle image selection
fileUploader.addEventListener("change", function (event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            cropTarget.src = e.target.result;
            showCropModal();
        };
        reader.readAsDataURL(file);
    } else {
        console.error("No file selected");
    }
});

// Show cropping modal
function showCropModal() {
    cropModal.style.display = "flex";
    setTimeout(() => {
        if (imageCropper) {
            imageCropper.destroy();
        }
        imageCropper = new Cropper(cropTarget, {
            viewMode: 2,
            autoCropArea: 1,
            movable: true,
            zoomable: true,
            rotatable: true,
            scalable: true,
            aspectRatio: NaN,
            center: true
        });
    }, 300);
}

// Close cropping modal
function hideCropModal() {
    cropModal.style.display = "none";
    if (imageCropper) {
        imageCropper.destroy();
        imageCropper = null;
    }
}

// Function to close the extracted ingredients display
function closeIngredientDisplay() {
    ingredientDisplay.style.display = "none";
    overlay.style.display = "none";
}

// Close extracted ingredients section when clicking overlay
overlay.addEventListener("click", closeIngredientDisplay);

// Handle image cropping and text extraction
confirmCrop.addEventListener('click', function () {
    if (imageCropper) {
        const croppedCanvas = imageCropper.getCroppedCanvas();
        if (croppedCanvas) {
            const croppedImageData = croppedCanvas.toDataURL("image/png");

            // Send the cropped image to the Flask backend
            fetch('/extract_text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: croppedImageData }),
            })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        // Clear previous content
                        ingredientList.innerHTML = '';

                        // Display the classification result
                        const classificationText = document.getElementById("classificationText");
                        classificationText.innerText = data.classification; // Set the classification text

                        // Display the ingredient list
                        data.text.forEach(ingredient => {
                            if (ingredient.trim() !== '') {
                                const li = document.createElement("li");
                                li.textContent = ingredient.trim();
                                ingredientList.appendChild(li);
                            }
                        });

                        // Show the ingredient section
                        overlay.style.display = "block";
                        ingredientDisplay.style.display = "block";
                    } else {
                        console.error("Error extracting text:", data.message);
                        alert("Failed to extract text. Please try again.");
                    }
                })
                .catch(error => {
                    console.error("Error:", error);
                    alert("An error occurred. Please check the console for details.");
                });

            hideCropModal();
        } else {
            console.error("Failed to crop image.");
            alert("Failed to crop image. Please try again.");
        }
    }
});


document.getElementById('allergenDetection').addEventListener('click', function() {
    // Redirect to the analysis page
    window.location.href = '/analysis';
});



