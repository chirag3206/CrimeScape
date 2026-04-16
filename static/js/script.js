document.addEventListener('DOMContentLoaded', () => {
    // 1. Tab Switching Logic
    const navLinks = document.querySelectorAll('.nav-links li');
    const tabContents = document.querySelectorAll('.tab-content');

    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            const target = link.getAttribute('data-tab');
            
            navLinks.forEach(l => l.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            link.classList.add('active');
            document.getElementById(target).classList.add('active');

            // Update Header
            const title = link.innerText.trim();
            document.getElementById('page-title').innerText = title;
            updateDescription(target);
        });
    });

    function updateDescription(tab) {
        const desc = document.getElementById('page-desc');
        if (tab === 'overview') desc.innerText = "Comprehensive crime intelligence and statistical modeling (2019-2023)";
        if (tab === 'ml-gallery') desc.innerText = "Visualization of state archetypes and dominant crime drivers";
        if (tab === 'predictor') desc.innerText = "Advanced ML projection center for future risk assessment";
    }

    // 2. Load Initial Stats
    fetch('/api/stats')
        .then(res => res.json())
        .then(data => {
            if (data.error) throw new Error(data.error);
            
            document.getElementById('worst-state').innerText = data.worst_state;
            document.getElementById('report-text').innerText = data.report_summary;
            
            // Populate State Dropdown
            const stateSelect = document.getElementById('state-select');
            stateSelect.innerHTML = '';
            data.states.forEach(s => {
                const opt = document.createElement('option');
                opt.value = opt.innerText = s;
                stateSelect.appendChild(opt);
            });
        })
        .catch(err => console.error("Stats Error:", err));

    // 3. Prediction Logic
    const predictBtn = document.getElementById('predict-btn');
    const resultSection = document.getElementById('prediction-result');

    predictBtn.addEventListener('click', () => {
        const state = document.getElementById('state-select').value;
        const domain = document.getElementById('domain-select').value;

        // Visual feedback
        predictBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        predictBtn.disabled = true;

        fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ state, domain })
        })
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                return;
            }

            const riskClass = data.risk_grade.includes('HIGH') ? 'risk-high' : 'risk-low';
            
            resultSection.innerHTML = `
                <div class="result-card anim-fade-in text-center">
                    <h2>Intelligence Profile for ${data.state}</h2>
                    <p class="subtext">Domain: ${data.domain}</p>
                    
                    <div class="result-val">${data.intensity_score}</div>
                    <p class="metric-title">Predicted Intensity Index (2026)</p>
                    <p class="definition">Expected crimes per 100,000 citizens based on projected 5-year growth trends.</p>

                    <div class="risk-badge ${riskClass}" style="margin-top: 2rem;">${data.risk_grade}</div>
                    <p class="metric-title">Regional Risk Grade</p>
                    <p class="definition">Classification relative to national safety benchmarks (Top 50% = High Risk).</p>

                    <p style="margin-top:1.5rem; font-size: 0.75rem; color: #94a3b8; opacity: 0.6;">Model Confidence: ${data.confidence}</p>
                </div>
            `;
        })
        .catch(err => {
            alert("Prediction failed. Ensure server is running.");
            console.error(err);
        })
        .finally(() => {
            predictBtn.innerHTML = 'Generate Forcast (2026)';
            predictBtn.disabled = false;
        });
    });

    // 4. Gallery Filter
    window.filterGallery = (category) => {
        const buttons = document.querySelectorAll('.gallery-controls button');
        buttons.forEach(b => b.classList.remove('active'));
        event.target.classList.add('active');

        const items = document.querySelectorAll('.gallery-item');
        items.forEach(item => {
            if (item.classList.contains(category)) {
                item.classList.remove('hidden');
            } else {
                item.classList.add('hidden');
            }
        });
    };

    // 5. REGIONAL MAP (LEADLET)
    let map, geojson;
    const mapContainer = document.getElementById('heatmap');
    
    // Full Mapping for GeoJSON name alignment (GeoJSON Name : My Data Name)
    const geoJSONToAPI = {
        "Andaman and Nicobar": "Andaman & Nicobar Islands",
        "Andhra Pradesh": "Andhra Pradesh",
        "Arunachal Pradesh": "Arunachal Pradesh",
        "Assam": "Assam",
        "Bihar": "Bihar",
        "Chandigarh": "Chandigarh",
        "Chhattisgarh": "Chhattisgarh",
        "Dadra and Nagar Haveli": "Dadra & Nagar Haveli and Daman & Diu",
        "Daman and Diu": "Dadra & Nagar Haveli and Daman & Diu",
        "Delhi": "Delhi (UT)",
        "Goa": "Goa",
        "Gujarat": "Gujarat",
        "Haryana": "Haryana",
        "Himachal Pradesh": "Himachal Pradesh",
        "Jammu and Kashmir": "Jammu & Kashmir", 
        "Jharkhand": "Jharkhand",
        "Karnataka": "Karnataka",
        "Kerala": "Kerala",
        "Lakshadweep": "Lakshadweep",
        "Madhya Pradesh": "Madhya Pradesh",
        "Maharashtra": "Maharashtra",
        "Manipur": "Manipur",
        "Meghalaya": "Meghalaya",
        "Mizoram": "Mizoram",
        "Nagaland": "Nagaland",
        "Orissa": "Odisha",
        "Puducherry": "Puducherry",
        "Punjab": "Punjab",
        "Rajasthan": "Rajasthan",
        "Sikkim": "Sikkim",
        "Tamil Nadu": "Tamil Nadu",
        "Tripura": "Tripura",
        "Uttar Pradesh": "Uttar Pradesh",
        "Uttaranchal": "Uttarakhand",
        "West Bengal": "West Bengal"
    };

    function initMap() {
        if (map) return;
        map = L.map('heatmap').setView([20.5937, 78.9629], 5);
        
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; CartoDB'
        }).addTo(map);

        updateMap();
    }

    function getColor(d, max) {
        const ratio = d / max;
        return ratio > 0.8 ? '#800026' :
               ratio > 0.6 ? '#BD0026' :
               ratio > 0.4 ? '#E31A1C' :
               ratio > 0.2 ? '#FC4E2A' :
               '#FED976';
    }

    function updateMap() {
        const domain = document.getElementById('map-domain-select').value;
        
        // Fetch intensity data for all states
        Promise.all([
            fetch(`/api/map_data/${domain}`).then(res => res.json()),
            fetch('https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson').then(res => res.json())
        ]).then(([data, geoData]) => {
            if (geojson) map.removeLayer(geojson);
            
            const maxVal = Math.max(...Object.values(data));

            geojson = L.geoJson(geoData, {
                style: (feature) => {
                    const geoName = feature.properties.NAME_1;
                    const apiName = geoJSONToAPI[geoName] || geoName;
                    const val = data[apiName] || 0;
                    return {
                        fillColor: getColor(val, maxVal),
                        weight: 1,
                        opacity: 1,
                        color: 'white',
                        fillOpacity: 0.7
                    };
                },
                onEachFeature: (feature, layer) => {
                    const geoName = feature.properties.NAME_1;
                    const apiName = geoJSONToAPI[geoName] || geoName;
                    const val = data[apiName] || 0;
                    layer.bindPopup(`
                        <div style="font-family: 'Outfit', sans-serif;">
                            <b style="font-size: 1.1rem; color: #1e293b;">${apiName}</b><br>
                            <span style="color: #3b82f6; font-weight: bold; font-size: 1.2rem;">${val}</span><br>
                            <small style="color: #64748b;">Predicted Intensity (Crimes per 100k citizens in 2026)</small>
                        </div>
                    `);
                }
            }).addTo(map);
        });
    }

    document.getElementById('map-domain-select').addEventListener('change', updateMap);
    
    // Trigger map load when tab is clicked
    document.querySelector('[data-tab="regional-map"]').addEventListener('click', () => {
        setTimeout(initMap, 200);
    });


    // 6. COMPARISON LAB
    const compBtn = document.getElementById('run-comparison');
    const compResults = document.getElementById('comp-results');

    // Populate Comparison Dropdowns
    fetch('/api/stats')
        .then(res => res.json())
        .then(data => {
            const dds = [document.getElementById('comp-state-1'), document.getElementById('comp-state-2')];
            dds.forEach(dd => {
                dd.innerHTML = '';
                data.states.forEach(s => {
                    const opt = document.createElement('option');
                    opt.value = opt.innerText = s;
                    dd.appendChild(opt);
                });
            });
        });

    compBtn.addEventListener('click', () => {
        const state1 = document.getElementById('comp-state-1').value;
        const state2 = document.getElementById('comp-state-2').value;
        const domain = document.getElementById('comp-domain-select').value;

        compBtn.innerHTML = '<i class="fas fa-microchip fa-spin"></i> Analyzing...';

        fetch('/api/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ state1, state2, domain })
        })
        .then(res => res.json())
        .then(data => {
            const card = (s, num) => `
                <div class="stat-card glass result-card anim-fade-in">
                    <h3>${s.name} Analysis</h3>
                    <div class="result-val">${s.intensity}</div>
                    <p class="metric-title">Intensity Index</p>
                    <small class="definition">Crimes per 100k (2026 Goal)</small>

                    <div class="risk-badge ${s.risk === 'HIGH RISK' ? 'risk-high' : 'risk-low'}" style="margin-top:1.5rem;">${s.risk}</div>
                    <p class="metric-title">Risk Rating</p>
                    <small class="definition">Vs National Median</small>
                </div>
            `;
            
            compResults.innerHTML = card(data.state1, 1) + card(data.state2, 2);
        })
        .finally(() => {
            compBtn.innerHTML = 'Initialize Head-to-Head Analysis';
        });
    });
});
