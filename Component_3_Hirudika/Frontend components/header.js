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

// Open file input when clicking "Use Photo"
document.querySelector(".upload-btn").addEventListener("click", function() {
    fileUploader.click();
});

// Handle image selection
fileUploader.addEventListener("change", function(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
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

// Crop image and display it inside the ingredient section
confirmCrop.addEventListener('click', function() {
    if (imageCropper) {
        const croppedCanvas = imageCropper.getCroppedCanvas();
        if (croppedCanvas) {
            const croppedImageData = croppedCanvas.toDataURL("image/png");

            // Append cropped image inside the ingredient display section
            const ingredientImageContainer = document.getElementById("ingredientDisplay");
            const newCroppedImage = document.createElement("img");
            newCroppedImage.src = croppedImageData;
            newCroppedImage.style.maxWidth = "100%";
            newCroppedImage.style.marginTop = "20px";

            // Remove previous images to avoid duplicates
            const existingImages = ingredientImageContainer.querySelectorAll("img");
            existingImages.forEach(img => img.remove());

            ingredientImageContainer.appendChild(newCroppedImage);

            // Show the ingredient section properly
            overlay.style.display = "block";
            ingredientDisplay.style.display = "block";

            hideCropModal();
        } else {
            console.error("Failed to crop image.");
        }
    }
});

// Function to close the extracted ingredients display
function closeIngredientDisplay() {
    ingredientDisplay.style.display = "none";
    overlay.style.display = "none";
}

// Close extracted ingredients section when clicking overlay
overlay.addEventListener("click", closeIngredientDisplay);





