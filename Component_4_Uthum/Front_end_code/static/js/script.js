document.addEventListener('DOMContentLoaded', function() {
    // Minimal JS - just for basic interactivity
    // No ingredient handling needed since this is done in the backend
    
    // Could add simple things like smooth scrolling or collapsible sections
    const toggleButtons = document.querySelectorAll('.toggle-section');
    if (toggleButtons) {
        toggleButtons.forEach(btn => {
            btn.addEventListener('click', function() {
                const targetId = this.getAttribute('data-target');
                const targetSection = document.getElementById(targetId);
                if (targetSection) {
                    targetSection.classList.toggle('hidden');
                }
            });
        });
    }
});