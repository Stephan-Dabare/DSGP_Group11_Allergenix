document.addEventListener("DOMContentLoaded", function() {
    const sections = document.querySelectorAll("section");
    const options = { threshold: 0.3 };

    let observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add("animate-slide-left");
            }
        });
    }, options);

    sections.forEach(section => observer.observe(section));

    // Update file name on file input change
    const fileInput = document.getElementById("file");
    const fileNameSpan = document.getElementById("file-name");

    fileInput.addEventListener("change", function() {
        if (fileInput.files.length > 0) {
            fileNameSpan.textContent = fileInput.files[0].name;
        } else {
            fileNameSpan.textContent = "No file chosen";
        }
    });
});