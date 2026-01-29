// Global variables
let predictionCount = 0;

// Sample data for testing
const sampleData = {
    1: {
        'Item_Weight': 8.5,
        'Item_Fat_Content': 'Low Fat',
        'Item_Visibility': 0.016,
        'Item_Type': 'Dairy',
        'Item_MRP': 249.81,
        'Outlet_Identifier': 'OUT013',
        'Outlet_Establishment_Year': 1987,
        'Outlet_Size': 'Medium',
        'Outlet_Location_Type': 'Tier 1',
        'Outlet_Type': 'Supermarket Type1'
    },
    2: {
        'Item_Weight': 15.2,
        'Item_Fat_Content': 'Regular',
        'Item_Visibility': 0.034,
        'Item_Type': 'Snack Foods',
        'Item_MRP': 89.95,
        'Outlet_Identifier': 'OUT027',
        'Outlet_Establishment_Year': 1985,
        'Outlet_Size': 'Small',
        'Outlet_Location_Type': 'Tier 2',
        'Outlet_Type': 'Grocery Store'
    },
    3: {
        'Item_Weight': 12.8,
        'Item_Fat_Content': 'Low Fat',
        'Item_Visibility': 0.028,
        'Item_Type': 'Household',
        'Item_MRP': 199.50,
        'Outlet_Identifier': 'OUT045',
        'Outlet_Establishment_Year': 2002,
        'Outlet_Size': 'High',
        'Outlet_Location_Type': 'Tier 3',
        'Outlet_Type': 'Supermarket Type2'
    }
};

// DOM elements
const form = document.getElementById('prediction-form');
const clearBtn = document.getElementById('clear-btn');
const sampleBtn = document.getElementById('sample-btn');
const loadingOverlay = document.getElementById('loading-overlay');
const predictBtn = document.getElementById('predict-btn');
const statusText = document.getElementById('status-text');
const countText = document.getElementById('count-text');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('BigMart Sales Prediction App Initialized');
    
    // Check model status
    checkModelStatus();
    
    // Add event listeners
    setupEventListeners();
    
    // Add form animations
    addFormAnimations();
    
    // Update prediction count from localStorage
    updatePredictionCount();
    
    // Load sample predictions
    loadSamplePredictions();
});

// Event listeners setup
function setupEventListeners() {
    // Form submission
    if (form) {
        form.addEventListener('submit', handleFormSubmit);
    }
    
    // Clear form button
    if (clearBtn) {
        clearBtn.addEventListener('click', clearForm);
    }
    
    // Sample data button
    if (sampleBtn) {
        sampleBtn.addEventListener('click', loadRandomSampleData);
    }
    
    // Input validation
    addInputValidation();
    
    // Auto-save form data
    addAutoSave();
}

// Handle form submission
function handleFormSubmit(e) {
    // Show loading overlay
    showLoading();
    
    // Increment prediction count
    predictionCount++;
    updatePredictionCount();
    
    // Add smooth scroll to results after form submission
    setTimeout(() => {
        const resultsSection = document.querySelector('.results-section');
        if (resultsSection) {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }, 100);
}

// Show loading overlay
function showLoading() {
    if (loadingOverlay) {
        loadingOverlay.style.display = 'flex';
        loadingOverlay.classList.add('fade-in');
    }
    
    if (predictBtn) {
        predictBtn.disabled = true;
        predictBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    }
}

// Hide loading overlay
function hideLoading() {
    if (loadingOverlay) {
        loadingOverlay.style.display = 'none';
        loadingOverlay.classList.remove('fade-in');
    }
    
    if (predictBtn) {
        predictBtn.disabled = false;
        predictBtn.innerHTML = '<i class="fas fa-magic"></i> Predict Sales';
    }
}

// Clear form
function clearForm() {
    if (form) {
        form.reset();
        
        // Clear any validation states
        const inputs = form.querySelectorAll('input, select');
        inputs.forEach(input => {
            input.classList.remove('invalid', 'valid');
        });
        
        // Clear localStorage
        localStorage.removeItem('bigmart_form_data');
        
        showNotification('Form cleared successfully!', 'info');
    }
}

// Load sample data
function loadSampleData(sampleNumber) {
    const data = sampleData[sampleNumber];
    if (!data) return;
    
    // Fill form with sample data
    Object.keys(data).forEach(key => {
        const input = document.querySelector(`[name="${key}"]`);
        if (input) {
            input.value = data[key];
            
            // Trigger change event for validation
            input.dispatchEvent(new Event('change'));
        }
    });
    
    showNotification(`Sample data ${sampleNumber} loaded!`, 'success');
    
    // Scroll to form
    const formSection = document.querySelector('.form-section');
    if (formSection) {
        formSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// Load random sample data
function loadRandomSampleData() {
    const randomSample = Math.floor(Math.random() * 3) + 1;
    loadSampleData(randomSample);
}

// Check model status
async function checkModelStatus() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        const modelStatus = document.getElementById('model-status');
        
        if (data.model_loaded) {
            statusText.textContent = 'Online';
            statusText.className = 'status-online';
            if (modelStatus) {
                modelStatus.classList.add('pulse');
            }
        } else {
            statusText.textContent = 'Offline';
            statusText.className = 'status-offline';
            showNotification('Model is not loaded. Some features may not work.', 'warning');
        }
    } catch (error) {
        console.error('Error checking model status:', error);
        statusText.textContent = 'Error';
        statusText.className = 'status-offline';
    }
}

// Load sample predictions
async function loadSamplePredictions() {
    try {
        const response = await fetch('/test_prediction');
        const data = await response.json();
        
        if (data.success) {
            updateSampleCards(data.test_results);
        }
    } catch (error) {
        console.error('Error loading sample predictions:', error);
    }
}

// Update sample cards with predictions
function updateSampleCards(results) {
    const sampleCards = document.querySelectorAll('.sample-card');
    
    results.forEach((result, index) => {
        if (sampleCards[index]) {
            const predictionSpan = sampleCards[index].querySelector('.sample-prediction');
            if (predictionSpan) {
                predictionSpan.textContent = result.prediction;
                predictionSpan.style.color = 'var(--success-color)';
                predictionSpan.style.fontWeight = '700';
            }
        }
    });
}

// Add input validation
function addInputValidation() {
    const inputs = document.querySelectorAll('input[type="number"]');
    
    inputs.forEach(input => {
        input.addEventListener('input', function() {
            validateInput(this);
        });
        
        input.addEventListener('blur', function() {
            validateInput(this);
        });
    });
    
    // Special validation for specific fields
    const itemVisibility = document.getElementById('Item_Visibility');
    if (itemVisibility) {
        itemVisibility.addEventListener('input', function() {
            const value = parseFloat(this.value);
            if (value < 0 || value > 1) {
                this.setCustomValidity('Visibility must be between 0 and 1');
            } else {
                this.setCustomValidity('');
            }
        });
    }
    
    const itemWeight = document.getElementById('Item_Weight');
    if (itemWeight) {
        itemWeight.addEventListener('input', function() {
            const value = parseFloat(this.value);
            if (value < 0) {
                this.setCustomValidity('Weight cannot be negative');
            } else {
                this.setCustomValidity('');
            }
        });
    }
    
    const itemMRP = document.getElementById('Item_MRP');
    if (itemMRP) {
        itemMRP.addEventListener('input', function() {
            const value = parseFloat(this.value);
            if (value < 0) {
                this.setCustomValidity('Price cannot be negative');
            } else {
                this.setCustomValidity('');
            }
        });
    }
}

// Validate individual input
function validateInput(input) {
    if (input.checkValidity()) {
        input.classList.remove('invalid');
        input.classList.add('valid');
    } else {
        input.classList.remove('valid');
        input.classList.add('invalid');
    }
}

// Add auto-save functionality
function addAutoSave() {
    const inputs = document.querySelectorAll('input, select');
    
    inputs.forEach(input => {
        input.addEventListener('change', saveFormData);
    });
    
    // Load saved data on page load
    loadFormData();
}

// Save form data to localStorage
function saveFormData() {
    const formData = new FormData(form);
    const data = {};
    
    for (let [key, value] of formData.entries()) {
        data[key] = value;
    }
    
    localStorage.setItem('bigmart_form_data', JSON.stringify(data));
}

// Load form data from localStorage
function loadFormData() {
    const savedData = localStorage.getItem('bigmart_form_data');
    
    if (savedData) {
        try {
            const data = JSON.parse(savedData);
            
            Object.keys(data).forEach(key => {
                const input = document.querySelector(`[name="${key}"]`);
                if (input && data[key]) {
                    input.value = data[key];
                }
            });
        } catch (error) {
            console.error('Error loading saved form data:', error);
        }
    }
}

// Update prediction count
function updatePredictionCount() {
    // Get count from localStorage
    const savedCount = localStorage.getItem('bigmart_prediction_count');
    if (savedCount) {
        predictionCount = parseInt(savedCount);
    }
    
    // Update display
    if (countText) {
        countText.textContent = predictionCount;
    }
    
    // Save updated count
    localStorage.setItem('bigmart_prediction_count', predictionCount.toString());
}

// Add form animations
function addFormAnimations() {
    const formGroups = document.querySelectorAll('.form-group');
    
    formGroups.forEach((group, index) => {
        group.style.animationDelay = `${index * 0.1}s`;
        group.classList.add('fade-in');
    });
    
    const statusCards = document.querySelectorAll('.status-card');
    statusCards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.2}s`;
        card.classList.add('slide-in');
    });
}

// Show notification
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas fa-${getNotificationIcon(type)}"></i>
        <span>${message}</span>
        <button class="notification-close" onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}

// Get notification icon based on type
function getNotificationIcon(type) {
    const icons = {
        'success': 'check-circle',
        'error': 'exclamation-triangle',
        'warning': 'exclamation-circle',
        'info': 'info-circle'
    };
    return icons[type] || 'info-circle';
}

// Real-time prediction (optional feature)
async function makeRealTimePrediction() {
    const formData = new FormData(form);
    const data = {};
    
    // Check if all required fields are filled
    const requiredFields = ['Item_MRP', 'Item_Type', 'Outlet_Type'];
    let allRequiredFilled = true;
    
    for (let field of requiredFields) {
        const value = formData.get(field);
        if (!value || value.trim() === '') {
            allRequiredFilled = false;
            break;
        }
        data[field] = value;
    }
    
    if (!allRequiredFilled) {
        return;
    }
    
    // Get all form data
    for (let [key, value] of formData.entries()) {
        data[key] = value;
    }
    
    try {
        const response = await fetch('/predict_api', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (result.success) {
            showQuickPreview(result.prediction);
        }
    } catch (error) {
        console.error('Real-time prediction error:', error);
    }
}

// Show quick preview of prediction
function showQuickPreview(prediction) {
    // Remove existing preview
    const existingPreview = document.querySelector('.quick-preview');
    if (existingPreview) {
        existingPreview.remove();
    }
    
    // Create preview element
    const preview = document.createElement('div');
    preview.className = 'quick-preview';
    preview.innerHTML = `
        <div class="preview-content">
            <i class="fas fa-eye"></i>
            <span>Quick Preview: ${prediction}</span>
            <button onclick="this.parentElement.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
    
    // Add to form
    const formActions = document.querySelector('.form-actions');
    if (formActions) {
        formActions.insertBefore(preview, formActions.firstChild);
    }
}

// Add CSS for notifications and quick preview
const additionalCSS = `
.notification {
    position: fixed;
    top: 20px;
    right: 20px;
    background: white;
    padding: 15px 20px;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    display: flex;
    align-items: center;
    gap: 10px;
    z-index: 1001;
    min-width: 300px;
    transform: translateX(100%);
    animation: slideInNotification 0.3s ease-out forwards;
}

.notification-success {
    border-left: 4px solid #10b981;
    background: linear-gradient(135deg, #ecfdf5, #f0fdf4);
}

.notification-error {
    border-left: 4px solid var(--error-color);
}

.notification-warning {
    border-left: 4px solid var(--warning-color);
}

.notification-info {
    border-left: 4px solid var(--info-color);
}

.notification i {
    font-size: 1.2rem;
}

.notification-success i {
    color: #10b981;
}

.notification-error i {
    color: var(--error-color);
}

.notification-warning i {
    color: var(--warning-color);
}

.notification-info i {
    color: var(--info-color);
}

.notification-close {
    background: none;
    border: none;
    cursor: pointer;
    padding: 5px;
    margin-left: auto;
    color: var(--text-light);
}

.notification-close:hover {
    color: var(--text-primary);
}

.quick-preview {
    margin-bottom: 15px;
    padding: 12px;
    background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
    border-radius: 8px;
    border: 1px solid var(--info-color);
}

.preview-content {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.9rem;
    color: var(--info-color);
    font-weight: 600;
}

.preview-content button {
    background: none;
    border: none;
    cursor: pointer;
    margin-left: auto;
    color: var(--info-color);
    padding: 2px;
}

@keyframes slideInNotification {
    to {
        transform: translateX(0);
    }
}

.form-group input.valid {
    border-color: var(--success-color);
}

.form-group input.invalid {
    border-color: var(--error-color);
}
`;

// Add the additional CSS to the page
const style = document.createElement('style');
style.textContent = additionalCSS;
document.head.appendChild(style);

// Hide loading overlay when page loads
window.addEventListener('load', () => {
    hideLoading();
});

// Add smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Console welcome message
console.log(`
🎯 BigMart Sales Prediction System
📊 Advanced ML-powered sales forecasting
🚀 Built with Flask, Scikit-learn & Interactive UI
💡 Made with ❤️ for accurate business predictions
`);
