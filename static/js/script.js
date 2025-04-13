document.addEventListener("DOMContentLoaded", () => {
    fetch("/static/data/emotion_logs.csv")
      .then(response => response.text())
      .then(csv => {
        Papa.parse(csv, {
          skipEmptyLines: true,
          complete: function(results) {
            const data = results.data;
            if (!data.length) return;
  
            const latestMap = new Map(); // key: email + modality
  
            // Get latest emotion entry for each (email, modality)
            data.forEach(row => {
              if (row.length < 5) return;
  
              const [authTime, email, modality, emotion, timestamp] = row;
              const key = `${email}_${modality}`;
  
              if (!latestMap.has(key) || new Date(timestamp) > new Date(latestMap.get(key).timestamp)) {
                latestMap.set(key, { authTime, email, modality, emotion, timestamp });
              }
            });
  
            const table = document.getElementById("employee-emotion-table");
            const alertSection = document.getElementById("alert-section");
            const recommendationSection = document.getElementById("recommendation-section");
  
            latestMap.forEach(({ authTime, email, modality, emotion, timestamp }) => {
              // Add to table
              const row = `<tr>
                <td>${authTime}</td>
                <td>${email}</td>
                <td>${modality}</td>
                <td>${emotion}</td>
              </tr>`;
              table.innerHTML += row;
  
              // Simple emotion-based alert
              const negativeEmotions = ["Sad", "Angry", "Disgust", "Fear"];
              if (negativeEmotions.includes(emotion)) {
                alertSection.innerHTML += `
                  <div class="alert">
                    <strong>${email}</strong> seems to be <strong>${emotion}</strong> based on <strong>${modality}</strong> input.
                  </div>`;
              } else {
                recommendationSection.innerHTML += `
                  <div class="recommendation">
                    <strong>${email}</strong> is <strong>${emotion}</strong>. Consider assigning productivity-boosting tasks.
                  </div>`;
              }
            });
          }
        });
      })
      .catch(error => {
        console.error("Error loading CSV:", error);
      });
  });
  