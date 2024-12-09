document.addEventListener('DOMContentLoaded', () => {
    // Form validation
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', (e) => {
            // Total number validation
            const totalClassRooms = document.getElementById('total_class_rooms');
            const totalTeachers = document.getElementById('total_teachers');
            const totalStudents = document.getElementById('total_students');

            if (parseInt(totalClassRooms.value) < 0 || 
                parseInt(totalTeachers.value) < 0 || 
                parseInt(totalStudents.value) < 0) {
                e.preventDefault();
                alert('Please enter non-negative values for classrooms, teachers, and students.');
                return;
            }

            // Grade configuration validation
            const gradeConfig = document.getElementById('grade_configuration').value;
            const gradeConfigRegex = /^\d+,\d+$/;
            if (!gradeConfigRegex.test(gradeConfig)) {
                e.preventDefault();
                alert('Grade configuration must be in the format "start,end" (e.g., "1,12")');
                return;
            }
        });
    }

    // Checkbox group interactivity (optional enhancement)
    const checkboxGroups = document.querySelectorAll('.checkbox-group');
    checkboxGroups.forEach(group => {
        const checkbox = group.querySelector('input[type="checkbox"]');
        const label = group.querySelector('label');

        label.addEventListener('click', () => {
            checkbox.checked = !checkbox.checked;
        });
    });

    // Print report functionality (for result page)
    const printBtn = document.querySelector('.btn-print');
    if (printBtn) {
        printBtn.addEventListener('click', () => {
            window.print();
        });
    }
});