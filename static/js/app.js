// Fraud Detection System - JavaScript Functionality

// Global variables
let currentResult = null;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeForm();
    setupEventListeners();
    initializeValidation();
    initializeTheme();
});

// Initialize form with validation
function initializeForm() {
    // Add input validation
    const inputs = document.querySelectorAll('input[type="number"]');
    inputs.forEach(input => {
        input.addEventListener('input', validateInput);
        input.addEventListener('blur', formatInput);
    });
}

// Initialize real-time validation
function initializeValidation() {
    const inputs = document.querySelectorAll('input[data-validation]');

    inputs.forEach(input => {
        // Real-time validation on input
        input.addEventListener('input', function() {
            validateField(this);
        });

        // Validation on blur
        input.addEventListener('blur', function() {
            validateField(this);
        });

        // Initial validation state
        validateField(input);
    });
}

// Validate individual field
function validateField(input) {
    const value = input.value;
    const validationType = input.getAttribute('data-validation');
    const min = parseFloat(input.getAttribute('min')) || 0;
    const max = parseFloat(input.getAttribute('max'));

    let isValid = true;
    let errorMessage = '';

    // Check if field is empty (but not required yet)
    if (value === '') {
        isValid = false;
        errorMessage = 'This field is required';
    } else {
        const numValue = parseFloat(value);

        // Check if it's a valid number
        if (isNaN(numValue)) {
            isValid = false;
            errorMessage = 'Please enter a valid number';
        } else {
            // Check range
            if (numValue < min) {
                isValid = false;
                errorMessage = `Value must be at least ${min}`;
            } else if (max && numValue > max) {
                isValid = false;
                errorMessage = `Value must be at most ${max}`;
            } else {
                // Custom validation based on type
                if (validationType === 'time') {
                    if (numValue < 0) {
                        isValid = false;
                        errorMessage = 'Time cannot be negative';
                    }
                } else if (validationType === 'amount') {
                    if (numValue < 0) {
                        isValid = false;
                        errorMessage = 'Amount cannot be negative';
                    } else if (numValue > 1000000) {
                        isValid = false;
                        errorMessage = 'Amount cannot exceed $1,000,000';
                    }
                }
            }
        }
    }

    // Update field appearance
    updateFieldValidation(input, isValid, errorMessage);

    return isValid;
}

// Update field validation appearance
function updateFieldValidation(input, isValid, errorMessage) {
    const container = input.closest('.form-group');
    const messageDiv = container.querySelector('.validation-message');

    // Remove previous validation classes
    input.classList.remove('field-valid', 'field-invalid');

    if (input.value !== '') {
        if (isValid) {
            input.classList.add('field-valid');
        } else {
            input.classList.add('field-invalid');
            if (messageDiv) {
                messageDiv.textContent = errorMessage;
                messageDiv.style.color = 'var(--danger-color)';
            }
        }
    } else {
        // Reset when empty
        if (messageDiv) {
            messageDiv.textContent = '';
        }
    }
}

// Validate all fields before form submission
function validateAllFields() {
    const inputs = document.querySelectorAll('input[data-validation]');
    let allValid = true;

    inputs.forEach(input => {
        if (!validateField(input)) {
            allValid = false;
        }
    });

    return allValid;
}

// Setup event listeners
function setupEventListeners() {
    // Form submission is handled by the form action
    // Additional JavaScript functionality can be added here
}

// Validate numeric input
function validateInput(event) {
    const input = event.target;
    const value = input.value;

    // Remove any non-numeric characters except decimal point
    const cleanValue = value.replace(/[^0-9.-]/g, '');

    // Ensure only one decimal point
    const parts = cleanValue.split('.');
    if (parts.length > 2) {
        input.value = parts[0] + '.' + parts.slice(1).join('');
    } else {
        input.value = cleanValue;
    }
}

// Format input on blur
function formatInput(event) {
    const input = event.target;
    const value = parseFloat(input.value);

    if (!isNaN(value)) {
        // Format to appropriate decimal places
        if (input.id === 'Time' || input.id === 'Amount') {
            input.value = value.toFixed(2);
        } else {
            input.value = value.toFixed(6);
        }
    }
}

// Clear all form inputs
function clearForm() {
    const inputs = document.querySelectorAll('input[type="number"]');
    inputs.forEach(input => {
        input.value = '';
    });

    // Hide results if visible
    hideResults();
}

// Fill form with sample data
function fillSampleData() {
    // Sample transaction data (only user-visible fields)
    const sampleData = {
        'Time': '125.5',
        'Amount': '49.99'
    };

    // Fill form inputs
    Object.keys(sampleData).forEach(key => {
        const input = document.getElementById(key);
        if (input) {
            input.value = sampleData[key];
        }
    });

    // Hide results when loading sample data
    hideResults();
}

// Show results section
function showResults(result) {
    const resultsSection = document.getElementById('resultsSection');
    const resultCard = document.getElementById('resultCard');

    // Remove previous result classes
    resultCard.classList.remove('result-success', 'result-error');

    // Add appropriate class
    if (result.prediction === 1) {
        resultCard.classList.add('result-error');
    } else {
        resultCard.classList.add('result-success');
    }

    // Update result content
    updateResultDisplay(result);

    // Show results section
    resultsSection.style.display = 'block';

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Hide results section
function hideResults() {
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'none';
}

// Update result display with animation
function updateResultDisplay(result) {
    const resultCard = document.getElementById('resultCard');
    const resultIcon = document.getElementById('resultIcon');
    const resultTitle = document.getElementById('resultTitle');
    const resultDescription = document.getElementById('resultDescription');
    const resultDetails = document.getElementById('resultDetails');

    // Remove previous classes
    resultCard.classList.remove('result-success', 'result-error', 'result-loading');

    // Update icon and colors based on result
    if (result.prediction === 1) {
        resultCard.classList.add('result-error');
        resultIcon.style.background = 'var(--danger-color)';
        resultIcon.innerHTML = '<i class="fas fa-exclamation-triangle"></i>';
        resultTitle.textContent = 'FRAUDULENT TRANSACTION';
        resultTitle.style.color = 'var(--danger-color)';
        resultDescription.textContent = 'This transaction has been flagged as potentially fraudulent.';
    } else {
        resultCard.classList.add('result-success');
        resultIcon.style.background = 'var(--success-color)';
        resultIcon.innerHTML = '<i class="fas fa-check-circle"></i>';
        resultTitle.textContent = 'NORMAL TRANSACTION';
        resultTitle.style.color = 'var(--success-color)';
        resultDescription.textContent = 'This transaction appears to be legitimate.';
    }

    // Update detailed metrics
    document.getElementById('fraudProb').textContent = `${(result.fraud_probability * 100).toFixed(2)}%`;
    document.getElementById('normalProb').textContent = `${(result.normal_probability * 100).toFixed(2)}%`;

    // Update confidence bar
    const confidencePercentage = Math.max(result.fraud_probability, result.normal_probability) * 100;
    document.getElementById('confidenceFill').style.setProperty('--confidence-width', `${confidencePercentage}%`);

    // Show details with animation
    setTimeout(() => {
        resultDetails.style.display = 'block';
        animateConfidenceBar(confidencePercentage);
    }, 500);
}

// Animate confidence bar
function animateConfidenceBar(percentage) {
    const fillElement = document.getElementById('confidenceFill');
    fillElement.style.setProperty('--confidence-width', '0%');

    setTimeout(() => {
        fillElement.style.setProperty('--confidence-width', `${percentage}%`);
    }, 100);
}

// Show loading state
function showLoading() {
    const resultCard = document.getElementById('resultCard');
    const resultIcon = document.getElementById('resultIcon');
    const resultTitle = document.getElementById('resultTitle');
    const resultDescription = document.getElementById('resultDescription');
    const resultDetails = document.getElementById('resultDetails');

    // Reset classes
    resultCard.classList.remove('result-success', 'result-error');

    // Show loading state
    resultIcon.style.background = 'var(--info-color)';
    resultIcon.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
    resultTitle.textContent = 'Analyzing Transaction...';
    resultTitle.style.color = 'var(--text-primary)';
    resultDescription.textContent = 'Processing transaction data with machine learning...';

    // Hide details during loading
    resultDetails.style.display = 'none';

    // Show results section
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';
}

// Error handling
function showError(message) {
    const resultCard = document.getElementById('resultCard');
    const resultIcon = document.getElementById('resultIcon');
    const resultTitle = document.getElementById('resultTitle');
    const resultDescription = document.getElementById('resultDescription');

    // Reset classes
    resultCard.classList.remove('result-success', 'result-error');
    resultCard.classList.add('result-error');

    resultIcon.style.background = 'var(--danger-color)';
    resultIcon.innerHTML = '<i class="fas fa-times-circle"></i>';
    resultTitle.textContent = 'ANALYSIS FAILED';
    resultTitle.style.color = 'var(--danger-color)';
    resultDescription.textContent = message || 'An error occurred while analyzing the transaction.';

    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';
}



// Form validation before submission
function validateForm() {
    const inputs = document.querySelectorAll('input[required]');
    let isValid = true;

    inputs.forEach(input => {
        if (!input.value.trim()) {
            input.style.borderColor = 'var(--danger-color)';
            isValid = false;
        } else {
            input.style.borderColor = 'var(--border-color)';
        }
    });

    return isValid;
}

// Initialize theme on page load
function initializeTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    setTheme(savedTheme);

    // Add click handler for theme toggle
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }
}

// Toggle between light and dark themes
function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
}

// Set the theme
function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);

    // Update toggle button appearance
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        themeToggle.setAttribute('title', `Switch to ${theme === 'light' ? 'dark' : 'light'} mode`);
    }
}

// Add loading state to form submission
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');

    if (form) {
        form.addEventListener('submit', function(e) {
            if (!validateForm()) {
                e.preventDefault();
                showError('Please fill in all required fields.');
                return;
            }

            // Show loading state
            showLoading();

            // Disable submit button
            const submitBtn = form.querySelector('button[type="submit"]');
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';

            // Re-enable after 10 seconds (in case of error)
            setTimeout(() => {
                submitBtn.disabled = false;
                submitBtn.innerHTML = '<i class="fas fa-search"></i> Analyze Transaction';
            }, 10000);
        });
    }
});
