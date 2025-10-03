// DOM Elements
const fileInput = document.getElementById("fileInput");
const uploadZone = document.getElementById("uploadZone");
const uploadBtn = document.getElementById("uploadBtn");
const fileInfo = document.getElementById("fileInfo");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const output = document.getElementById("output");
const rawText = document.getElementById("rawText");
const reportImage = document.getElementById("reportImage");
const imagePlaceholder = document.getElementById("imagePlaceholder");
const evalOutput = document.getElementById("evalOutput");
const progressSection = document.getElementById("progressSection");
const progressFill = document.getElementById("progressFill");
const hitlForm = document.getElementById("hitlForm");
const saveBtn = document.getElementById("saveBtn");
const evaluateBtn = document.getElementById("evaluateBtn");
const toast = document.getElementById("toast");
const results = document.getElementById("results");
const evaluation = document.getElementById("evaluation");
const correctionSection = document.getElementById("correctionSection");

// Patient info elements
const patientName = document.getElementById("patientName");
const patientAge = document.getElementById("patientAge");
const patientSex = document.getElementById("patientSex");
const patientDate = document.getElementById("patientDate");
const testResults = document.getElementById("testResults");

// Evaluation metric elements
const regexAccuracy = document.getElementById("regexAccuracy");
const hybridAccuracy = document.getElementById("hybridAccuracy");
const totalFiles = document.getElementById("totalFiles");
const processedFiles = document.getElementById("processedFiles");

let currentFile = null;
let currentData = null;
const API_URL = window.location.origin;

// Toast notification system
function showToast(message) {
    const toastMessage = toast.querySelector('.toast-message');
    toastMessage.textContent = message;
    toast.className = 'toast show';
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// File size formatter
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Progress animation
function animateProgress(targetWidth) {
    progressFill.style.width = targetWidth + '%';
}

// Progress step animation
function updateProgressStep(stepNumber) {
    const steps = document.querySelectorAll('.step');
    steps.forEach((step, index) => {
        if (index < stepNumber) {
            step.classList.add('active');
        } else {
            step.classList.remove('active');
        }
    });
}

// File upload handling
function handleFileSelect(file) {
  if (!file) return;

    currentFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    
    // Show file info and hide upload zone
    uploadZone.style.display = 'none';
    fileInfo.style.display = 'flex';
    fileInfo.classList.add('fade-in');
    
    showToast(`File "${file.name}" selected successfully`);
}

// File input change handler
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    handleFileSelect(file);
});

// Drag and drop handlers
uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        fileInput.files = files;
        handleFileSelect(file);
    }
});

// Upload button handler
uploadBtn.addEventListener('click', async () => {
    if (!currentFile) {
        showToast('Please select a file first');
        return;
    }

    await uploadFile(currentFile);
});

// File upload function
async function uploadFile(file) {
    try {
        // Show progress section
        progressSection.style.display = 'block';
        animateProgress(25);
        
        const formData = new FormData();
        formData.append('file', file);
        
        // Simulate OCR step
        setTimeout(() => {
            animateProgress(50);
        }, 500);
        
        // Simulate AI extraction step
        setTimeout(() => {
            animateProgress(75);
        }, 1000);
        
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Upload failed');
        }
        
        const data = await response.json();
        
        // Complete progress
        animateProgress(100);
        
        setTimeout(() => {
            progressSection.style.display = 'none';
            displayResults(data);
        }, 500);
        
        showToast('File processed successfully');
        
    } catch (error) {
        progressSection.style.display = 'none';
        showToast('Upload failed');
    }
}

// Display results function
function displayResults(data) {
    currentData = data;
    
    // Show results section
    results.style.display = 'block';
    results.classList.add('slide-up');
    
    // Display raw text
    rawText.textContent = data.raw_text || 'No text extracted';
    
    // Display JSON
    output.textContent = JSON.stringify(data.extracted, null, 2);
    
    // Display patient information
    if (data.extracted && data.extracted.patient) {
        const patient = data.extracted.patient;
        patientName.textContent = patient.name || '-';
        patientAge.textContent = patient.age || '-';
        patientSex.textContent = patient.sex || '-';
        patientDate.textContent = patient.date || '-';
    }
    
    // Display test results
    if (data.extracted && data.extracted.tests) {
        displayTestResults(data.extracted.tests);
    }
    
    // Display annotated image
    if (data.image) {
        reportImage.src = data.image;
        reportImage.style.display = 'block';
        imagePlaceholder.style.display = 'none';
        reportImage.onload = () => {
            reportImage.classList.add('fade-in');
        };
    }
    
    // Show evaluation and correction sections
    evaluation.style.display = 'block';
    correctionSection.style.display = 'block';
    
    // Enable correction form
    renderEditableForm(data.extracted);
}

// Display test results
function displayTestResults(tests) {
    if (!tests || tests.length === 0) {
        testResults.innerHTML = '<p class="no-data">No test results found</p>';
        return;
    }
    
    const testsHtml = tests.map(test => `
        <div class="test-item">
            <div class="test-name">${test.name || 'Unknown Test'}</div>
            <div class="test-value">
                <span>${test.value || '-'} ${test.unit || ''}</span>
                ${test.confidence ? `<span class="confidence-badge">${(test.confidence * 100).toFixed(0)}%</span>` : ''}
            </div>
        </div>
    `).join('');
    
    testResults.innerHTML = testsHtml;
}

// Render editable form for corrections
function renderEditableForm(data) {
    if (!data) {
        hitlForm.innerHTML = '<p class="no-data">No data to correct</p>';
        saveBtn.disabled = true;
        return;
    }
    
    let html = '<div class="correction-form">';
    
    // Patient information
    html += '<h3>Patient Information</h3>';
    html += '<div>';
    
    for (const [key, value] of Object.entries(data.patient || {})) {
        html += '<div class="form-group">'
              + '<label>' + (key.charAt(0).toUpperCase() + key.slice(1)) + ':</label>'
              + '<input type="text" id="pat_' + key + '" value="' + (value || '') + '">'
              + '</div>';
    }
    html += '</div>';
    
    // Test results
    html += '<h3>Test Results</h3>';
    html += '<div>';
    
    (data.tests || []).forEach((test, index) => {
        html += '<div class="test-correction-item">'
              + '<div>'
              + '<label>Test Name:</label>'
              + '<input type="text" id="test_' + index + '_name" value="' + (test.name || '') + '">'
              + '</div>'
              + '<div>'
              + '<label>Value:</label>'
              + '<input type="text" id="test_' + index + '_value" value="' + (test.value || '') + '">'
              + '</div>'
              + '<div>'
              + '<label>Unit:</label>'
              + '<input type="text" id="test_' + index + '_unit" value="' + (test.unit || '') + '">'
              + '</div>'
              + '</div>';
    });
    
    html += '</div></div>';
    
    hitlForm.innerHTML = html;
    saveBtn.disabled = false;
}

// Save corrections
saveBtn.addEventListener('click', async () => {
    if (!currentData) {
        showToast('No data to save');
        return;
    }
    
    try {
        const correctedData = collectFormData();
        
        const response = await fetch(`${API_URL}/correct`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                filename: currentFile.name,
                corrected: correctedData
            })
        });
        
        if (!response.ok) {
            throw new Error('Save failed');
        }
        
        const result = await response.json();
        showToast('Corrections saved successfully');
        
    } catch (error) {
        showToast('Failed to save corrections');
    }
});

// Collect form data
function collectFormData() {
    const patient = {};
    const tests = [];
    
    // Collect patient data
    const patientInputs = document.querySelectorAll('[id^="pat_"]');
    patientInputs.forEach(input => {
        const key = input.id.replace('pat_', '');
        patient[key] = input.value;
    });
    
    // Collect test data
    const testInputs = document.querySelectorAll('[id^="test_"]');
    const testData = {};
    
    testInputs.forEach(input => {
        const parts = input.id.split('_');
        const index = parts[1];
        const field = parts[2];
        
        if (!testData[index]) {
            testData[index] = {};
        }
        testData[index][field] = input.value;
    });
    
    // Convert to array
    Object.values(testData).forEach(test => {
        tests.push(test);
    });
    
    return { patient, tests };
}

// Evaluation
evaluateBtn.addEventListener('click', async () => {
    try {
        evaluateBtn.disabled = true;
        evaluateBtn.textContent = 'Running...';
        
        const response = await fetch(`${API_URL}/evaluate`);
        
        if (!response.ok) {
            throw new Error('Eval failed');
        }
        
        const data = await response.json();
        displayEvaluationResults(data);
        showToast('Evaluation completed');
        
    } catch (error) {
        showToast('Evaluation failed');
    } finally {
        evaluateBtn.disabled = false;
        evaluateBtn.textContent = 'Run Evaluation';
    }
});

// Display evaluation results
function displayEvaluationResults(data) {
    regexAccuracy.textContent = data.regex_accuracy ? (data.regex_accuracy * 100).toFixed(1) + '%' : '-';
    hybridAccuracy.textContent = data.hybrid_accuracy ? (data.hybrid_accuracy * 100).toFixed(1) + '%' : '-';
    totalFiles.textContent = data.total_files || '-';
    processedFiles.textContent = data.processed_files || '-';
}

// Utility functions
function copyToClipboard(elementId) {
    const element = document.getElementById(elementId);
    navigator.clipboard.writeText(element.textContent).then(() => {
        showToast('Copied to clipboard!');
    }).catch(() => {
        showToast('Failed to copy to clipboard');
    });
}

function downloadJSON() {
    if (!currentData) {
        showToast('No data to download');
        return;
    }
    
    const dataStr = JSON.stringify(currentData.extracted, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${currentFile.name}_extracted.json`;
    link.click();
    URL.revokeObjectURL(url);
    showToast('JSON file downloaded!');
}

function downloadImage() {
    const img = document.getElementById('reportImage');
    if (!img.src) {
        showToast('No image to download');
        return;
    }
    
    const link = document.createElement('a');
    link.href = img.src;
    link.download = `${currentFile.name}_annotated.png`;
    link.click();
    showToast('Image downloaded!');
}

// Smooth scrolling for navigation links
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const targetId = link.getAttribute('href').substring(1);
        const targetElement = document.getElementById(targetId);
        if (targetElement) {
            targetElement.scrollIntoView({ behavior: 'smooth' });
        }
    });
});

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    showToast('Ready. Upload a lab report.');
});