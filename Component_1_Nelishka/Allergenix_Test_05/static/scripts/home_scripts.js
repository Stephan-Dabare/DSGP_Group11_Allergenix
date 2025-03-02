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
});