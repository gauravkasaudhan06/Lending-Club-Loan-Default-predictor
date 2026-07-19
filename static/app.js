// LendingClub Risk Predictor App JS

document.addEventListener("DOMContentLoaded", () => {
    // Slider values updates
    const intRateSlider = document.getElementById("int_rate");
    const intRateVal = document.getElementById("int_rate_val");
    
    intRateSlider.addEventListener("input", (e) => {
        intRateVal.textContent = `${e.target.value}%`;
    });
    
    const dtiSlider = document.getElementById("dti");
    const dtiVal = document.getElementById("dti_val");
    
    dtiSlider.addEventListener("input", (e) => {
        dtiVal.textContent = e.target.value;
    });

    // Form submission
    const form = document.getElementById("predictor-form");
    const submitBtn = document.getElementById("submit-btn");
    const btnLoader = document.getElementById("btn-loader");
    const submitBtnText = submitBtn.querySelector("span");
    
    const placeholderContent = document.getElementById("placeholder-content");
    const realContent = document.getElementById("real-content");
    
    // Result card indicators
    const progressCircle = document.getElementById("progress-circle");
    const riskVal = document.getElementById("risk-val");
    const recBanner = document.getElementById("rec-banner");
    const recTitle = document.getElementById("rec-title");
    const recDesc = document.getElementById("rec-desc");
    
    // Result table details
    const detailRec = document.getElementById("detail-rec");
    const detailProb = document.getElementById("detail-prob");
    const detailRepay = document.getElementById("detail-repay");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        // Show loading state
        submitBtn.disabled = true;
        btnLoader.style.display = "block";
        submitBtnText.textContent = "Processing Risk Profile...";
        
        // Gather values
        const payload = {
            loan_amnt: parseFloat(document.getElementById("loan_amnt").value),
            term: parseInt(document.getElementById("term").value),
            int_rate: parseFloat(document.getElementById("int_rate").value),
            installment: parseFloat(document.getElementById("installment").value),
            annual_inc: parseFloat(document.getElementById("annual_inc").value),
            dti: parseFloat(document.getElementById("dti").value),
            home_ownership: document.getElementById("home_ownership").value,
            earliest_cr_line: parseInt(document.getElementById("earliest_cr_line").value),
            open_acc: parseInt(document.getElementById("open_acc").value),
            total_acc: parseInt(document.getElementById("total_acc").value),
            mort_acc: parseInt(document.getElementById("mort_acc").value),
            sub_grade: document.getElementById("sub_grade").value,
            purpose: document.getElementById("purpose").value,
            verification_status: document.getElementById("verification_status").value,
            zip_code: document.getElementById("zip_code").value,
            
            // Hidden defaults
            pub_rec: parseInt(document.getElementById("pub_rec").value),
            pub_rec_bankruptcies: parseInt(document.getElementById("pub_rec_bankruptcies").value),
            revol_bal: parseFloat(document.getElementById("revol_bal").value),
            revol_util: parseFloat(document.getElementById("revol_util").value),
            initial_list_status: document.getElementById("initial_list_status").value,
            application_type: document.getElementById("application_type").value
        };

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });
            
            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.error || "Failed to contact prediction server.");
            }
            
            // Success: Switch placeholder card with report
            placeholderContent.classList.add("hidden");
            realContent.classList.remove("hidden");
            
            // Animate results
            animateResults(result);
            
        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            // Restore button state
            submitBtn.disabled = false;
            btnLoader.style.display = "none";
            submitBtnText.textContent = "Evaluate Default Risk";
        }
    });

    function animateResults(result) {
        const targetRisk = result.risk_percentage;
        const riskClass = result.risk_class;
        
        // Define colors and styles based on risk classes
        let gaugeColor = "#10b981"; // green
        let bannerClass = "rec-low";
        
        if (riskClass === "moderate") {
            gaugeColor = "#f59e0b"; // orange
            bannerClass = "rec-moderate";
        } else if (riskClass === "high") {
            gaugeColor = "#ef4444"; // red
            bannerClass = "rec-high";
        }
        
        // Set CSS custom property for color
        progressCircle.style.setProperty("--gauge-color", gaugeColor);
        
        // Set banner class and content
        recBanner.className = `recommendation-banner ${bannerClass}`;
        recTitle.textContent = result.recommendation;
        
        if (riskClass === "low") {
            recDesc.textContent = "Applicant displays strong financial health. The credit score, low DTI ratio, and repayment history suggest minimal default threat.";
        } else if (riskClass === "moderate") {
            recDesc.textContent = "Caution advised. Moderate debt loads, borderline DTI ratios, or mid-tier subgrades pose a mild default threat. Approve with higher interest rate limits.";
        } else {
            recDesc.textContent = "High danger of credit default. Extremely high debt burden (DTI), subgrade levels, or insufficient income indicators suggest significant threat of loss.";
        }
        
        // Load details table
        detailRec.textContent = result.recommendation;
        detailProb.textContent = (result.probability_default).toFixed(4);
        detailRepay.textContent = (1.0 - result.probability_default).toFixed(4);
        
        // Animate circular progress and counter
        let startVal = 0;
        const duration = 1200; // ms
        const startTime = performance.now();
        
        function updateGauge(currentTime) {
            const elapsedTime = currentTime - startTime;
            const progress = Math.min(elapsedTime / duration, 1);
            
            // Cubic ease-out
            const easeProgress = 1 - Math.pow(1 - progress, 3);
            const currentRiskVal = easeProgress * targetRisk;
            
            riskVal.textContent = `${currentRiskVal.toFixed(1)}%`;
            
            // Conic gradient rotation degrees (360 degrees total)
            const deg = (currentRiskVal / 100) * 360;
            progressCircle.style.background = `conic-gradient(${gaugeColor} ${deg}deg, rgba(255, 255, 255, 0.05) ${deg}deg)`;
            
            if (progress < 1) {
                requestAnimationFrame(updateGauge);
            }
        }
        
        requestAnimationFrame(updateGauge);
    }
});
