/**
 * Healthcare Patient Risk Prediction - Client Script
 * Handles disease selection, sample data population, form validation,
 * and AJAX interaction with the Flask prediction backend.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const diseaseCards = document.querySelectorAll('.disease-card');
    const diseaseInput = document.getElementById('selected-disease-input');
    const activeDiseaseTitle = document.getElementById('active-disease-title');
    const activeDiseaseIcon = document.getElementById('active-disease-icon');
    const activeDiseaseText = document.getElementById('active-disease-text');
    const predictionForm = document.getElementById('prediction-form');
    const predictBtn = document.getElementById('predict-submit-btn');
    const btnSpinner = document.getElementById('btn-spinner');
    const btnText = document.getElementById('btn-text');
    const errorBanner = document.getElementById('form-error-banner');
    
    // Result Card Elements
    const resultPlaceholder = document.getElementById('result-placeholder');
    const resultDisplay = document.getElementById('result-display');
    const resPatientName = document.getElementById('res-patient-name');
    const resDiseaseName = document.getElementById('res-disease-name');
    const resRiskBadge = document.getElementById('res-risk-badge');
    const resProbabilityBox = document.getElementById('res-probability-box');
    const resProbVal = document.getElementById('res-prob-val');
    const resProbBar = document.getElementById('res-prob-bar');
    const resSummaryText = document.getElementById('res-summary-text');
    const resAlgorithm = document.getElementById('res-algorithm');
    const resRawPrediction = document.getElementById('res-raw-prediction');
    const resStatusLabel = document.getElementById('res-status-label');
    const resDisclaimer = document.getElementById('res-disclaimer');

    // Quick Sample Buttons
    const btnSampleNormal = document.getElementById('btn-sample-normal');
    const btnSampleElevated = document.getElementById('btn-sample-elevated');

    // Disease UI Configurations
    const diseaseConfig = {
        diabetes: {
            title: "Diabetes Risk Assessment",
            icon: "🩸",
            normal: {
                Pregnancies: 1, Glucose: 85, BloodPressure: 66, SkinThickness: 29,
                Insulin: 0, BMI: 26.6, DiabetesPedigreeFunction: 0.35, Age: 31
            },
            elevated: {
                Pregnancies: 6, Glucose: 148, BloodPressure: 72, SkinThickness: 35,
                Insulin: 0, BMI: 33.6, DiabetesPedigreeFunction: 0.62, Age: 50
            }
        },
        heart: {
            title: "Heart Disease Risk Assessment",
            icon: "❤️",
            normal: {
                age: 67, sex: "1", cp: "0", trestbps: 160, chol: 286, fbs: "0",
                restecg: "0", thalach: 108, exang: "1", oldpeak: 1.5, slope: "1", ca: "3", thal: "2"
            },
            elevated: {
                age: 57, sex: "0", cp: "0", trestbps: 120, chol: 354, fbs: "0",
                restecg: "1", thalach: 163, exang: "1", oldpeak: 0.6, slope: "2", ca: "0", thal: "2"
            }
        },
        kidney: {
            title: "Chronic Kidney Disease Assessment",
            icon: "🫘",
            normal: {
                age: 40, bp: 80, sg: "1.025", al: "0", su: "0", bgr: 140, bu: 10,
                sc: 1.2, sod: 135, pot: 5.0, hemo: 15.0, pcv: 48, wc: 6200, rc: 5.3,
                rbc: "normal", pc: "normal", pcc: "notpresent", ba: "notpresent",
                htn: "no", dm: "no", cad: "no", appet: "good", pe: "no", ane: "no"
            },
            elevated: {
                age: 48, bp: 80, sg: "1.020", al: "1", su: "0", bgr: 121, bu: 36,
                sc: 1.2, sod: 138, pot: 4.4, hemo: 15.4, pcv: 44, wc: 7800, rc: 5.2,
                rbc: "normal", pc: "normal", pcc: "notpresent", ba: "notpresent",
                htn: "yes", dm: "yes", cad: "no", appet: "good", pe: "no", ane: "no"
            }
        },
        liver: {
            title: "Liver Disease Risk Assessment",
            icon: "🫀",
            normal: {
                Age: 35, Gender: "Male", Total_Bilirubin: 0.9, Direct_Bilirubin: 0.2,
                Alkaline_Phosphotase: 190, Alamine_Aminotransferase: 25, Aspartate_Aminotransferase: 20,
                Total_Protiens: 6.4, Albumin: 3.6, Albumin_and_Globulin_Ratio: 1.2
            },
            elevated: {
                Age: 38, Gender: "Male", Total_Bilirubin: 3.1, Direct_Bilirubin: 1.6,
                Alkaline_Phosphotase: 253, Alamine_Aminotransferase: 80, Aspartate_Aminotransferase: 406,
                Total_Protiens: 6.8, Albumin: 3.9, Albumin_and_Globulin_Ratio: 1.3
            }
        }
    };

    /**
     * Switch currently active disease module
     */
    function switchDisease(disease) {
        if (!diseaseConfig[disease]) return;

        // Update card active classes
        diseaseCards.forEach(card => {
            if (card.dataset.disease === disease) {
                card.classList.add('active');
            } else {
                card.classList.remove('active');
            }
        });

        // Update form hidden input
        diseaseInput.value = disease;

        // Update header
        activeDiseaseIcon.textContent = diseaseConfig[disease].icon;
        activeDiseaseText.textContent = diseaseConfig[disease].title;

        // Switch field container
        document.querySelectorAll('.disease-fields-group').forEach(group => {
            group.classList.remove('active');
        });
        const activeGroup = document.getElementById(`fields-${disease}`);
        if (activeGroup) {
            activeGroup.classList.add('active');
        }

        // Hide any previous error
        hideError();
    }

    // Attach click events to disease cards
    diseaseCards.forEach(card => {
        card.addEventListener('click', () => {
            const disease = card.dataset.disease;
            switchDisease(disease);
        });
    });

    /**
     * Fill form with sample values
     */
    function fillSample(type) {
        const currentDisease = diseaseInput.value;
        const config = diseaseConfig[currentDisease];
        if (!config || !config[type]) return;

        const data = config[type];
        const activeContainer = document.getElementById(`fields-${currentDisease}`);
        if (!activeContainer) return;

        Object.keys(data).forEach(fieldName => {
            const input = activeContainer.querySelector(`[name="${fieldName}"]`);
            if (input) {
                input.value = data[fieldName];
            }
        });

        // Set patient name recommendation if empty
        const nameInput = document.getElementById('patient_name');
        if (!nameInput.value.trim()) {
            nameInput.value = type === 'normal' ? 'Sample Patient (Normal Cohort)' : 'Sample Patient (Risk Cohort)';
        }

        hideError();
    }

    btnSampleNormal.addEventListener('click', () => fillSample('normal'));
    btnSampleElevated.addEventListener('click', () => fillSample('elevated'));

    /**
     * Error banner helpers
     */
    function showError(message) {
        errorBanner.textContent = message;
        errorBanner.classList.remove('hidden');
    }

    function hideError() {
        errorBanner.textContent = '';
        errorBanner.classList.add('hidden');
    }

    /**
     * Handle form submission
     */
    predictionForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        hideError();

        const currentDisease = diseaseInput.value;
        const activeContainer = document.getElementById(`fields-${currentDisease}`);
        if (!activeContainer) {
            showError("Selected disease form is unavailable.");
            return;
        }

        // Gather patient name (display only)
        const patientName = document.getElementById('patient_name').value.trim() || 'Anonymous Patient';

        // Gather all inputs from active container
        const payload = {
            disease: currentDisease,
            patient_name: patientName
        };

        const inputs = activeContainer.querySelectorAll('input, select');
        let hasEmptyFields = false;
        let invalidNumbers = false;

        inputs.forEach(input => {
            const val = input.value.trim();
            if (val === '') {
                hasEmptyFields = true;
            }

            if (input.type === 'number') {
                const num = parseFloat(val);
                if (isNaN(num)) {
                    invalidNumbers = true;
                }
                const min = input.getAttribute('min');
                const max = input.getAttribute('max');
                if (min !== null && num < parseFloat(min)) {
                    invalidNumbers = true;
                }
                if (max !== null && num > parseFloat(max)) {
                    invalidNumbers = true;
                }
                payload[input.name] = num;
            } else {
                payload[input.name] = val;
            }
        });

        if (hasEmptyFields) {
            showError("Please fill in all clinical parameters for this assessment.");
            return;
        }

        if (invalidNumbers) {
            showError("Please verify that all numerical values are valid and within clinical bounds.");
            return;
        }

        // Set Loading State
        predictBtn.disabled = true;
        btnSpinner.classList.remove('hidden');
        btnText.textContent = "Analyzing...";

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (!response.ok || data.status === 'error') {
                showError(data.message || "An error occurred during prediction analysis.");
                return;
            }

            // Render Result
            displayResult(data);

        } catch (err) {
            console.error("Prediction request failed:", err);
            showError("Network connection error. Please verify the Flask server is running locally.");
        } finally {
            // Restore button state
            predictBtn.disabled = false;
            btnSpinner.classList.add('hidden');
            btnText.textContent = "⚡ Predict Clinical Risk";
        }
    });

    /**
     * Render prediction result into result card
     */
    function displayResult(data) {
        // Toggle view
        resultPlaceholder.classList.add('hidden');
        resultDisplay.classList.remove('hidden');

        // Populate header details
        resPatientName.textContent = `Patient: ${data.patient_name}`;
        resDiseaseName.textContent = `Selected: ${data.disease_name}`;

        // Risk badge
        resRiskBadge.textContent = data.risk_level;
        resRiskBadge.className = `result-badge ${data.risk_class}`;

        // Probability display
        if (data.probability !== null && data.probability !== undefined) {
            resProbabilityBox.classList.remove('hidden');
            resProbVal.textContent = `${data.probability}%`;
            resProbBar.style.width = `${Math.min(Math.max(data.probability, 5), 100)}%`;
            resProbBar.className = `prob-bar-fill ${data.risk_class}`;
        } else {
            resProbabilityBox.classList.add('hidden');
        }

        // Summary and metadata
        resSummaryText.textContent = data.summary;
        resAlgorithm.textContent = data.algorithm;
        resRawPrediction.textContent = `Risk Class ${data.prediction} (${data.risk_class === 'risk-high' ? 'Positive' : 'Negative'})`;
        resStatusLabel.textContent = data.status_label || "Educational ML Risk Estimate";
        resDisclaimer.textContent = data.disclaimer || "Educational ML estimate only — not a medical diagnosis.";

        // On mobile, scroll smoothly to the result
        if (window.innerWidth < 992) {
            resultDisplay.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }
});
