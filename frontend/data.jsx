window.OCEANIQ_DATA = {
    overview: [
        { id: "Z-01", zone: "Chennai", rainfall: 312, sst: 30.1, ndvi: 0.12, discharge: 1450, rmnpi: 0.87, risk: "CRITICAL", lat: 13.0827, lon: 80.2707 },
        { id: "Z-02", zone: "Kolkata", rainfall: 420, sst: 29.5, ndvi: 0.34, discharge: 3200, rmnpi: 0.71, risk: "HIGH", lat: 22.5726, lon: 88.3639 },
        { id: "Z-03", zone: "Mumbai", rainfall: 580, sst: 29.8, ndvi: 0.22, discharge: 850, rmnpi: 0.68, risk: "HIGH", lat: 18.9667, lon: 72.8333 },
        { id: "Z-04", zone: "Kochi", rainfall: 610, sst: 28.9, ndvi: 0.65, discharge: 620, rmnpi: 0.44, risk: "MODERATE", lat: 9.9312, lon: 76.2673 },
        { id: "Z-05", zone: "Visakhapatnam", rainfall: 150, sst: 29.2, ndvi: 0.41, discharge: 400, rmnpi: 0.38, risk: "MODERATE", lat: 17.6868, lon: 83.2185 },
        { id: "Z-06", zone: "Mangalore", rainfall: 820, sst: 28.5, ndvi: 0.78, discharge: 910, rmnpi: 0.19, risk: "LOW", lat: 12.8688, lon: 74.8436 },
        { id: "Z-07", zone: "Puducherry", rainfall: 280, sst: 30.4, ndvi: 0.15, discharge: 110, rmnpi: 0.91, risk: "CRITICAL", lat: 11.9416, lon: 79.8083 },
        { id: "Z-08", zone: "Tuticorin", rainfall: 90, sst: 30.2, ndvi: 0.21, discharge: 80, rmnpi: 0.63, risk: "HIGH", lat: 8.7642, lon: 78.1348 },
    ],
    timeseries: [
        { month: "Jan", rainfall: 15, sst: 26.5, ndvi: 0.45, discharge: 300, anomalies: 0 },
        { month: "Feb", rainfall: 10, sst: 27.1, ndvi: 0.42, discharge: 280, anomalies: 0 },
        { month: "Mar", rainfall: 25, sst: 28.4, ndvi: 0.38, discharge: 250, anomalies: 0 },
        { month: "Apr", rainfall: 40, sst: 29.8, ndvi: 0.31, discharge: 220, anomalies: 1 },
        { month: "May", rainfall: 120, sst: 30.2, ndvi: 0.25, discharge: 450, anomalies: 4 },
        { month: "Jun", rainfall: 380, sst: 29.5, ndvi: 0.35, discharge: 1800, anomalies: 12 },
        { month: "Jul", rainfall: 450, sst: 28.8, ndvi: 0.55, discharge: 3200, anomalies: 15 },
        { month: "Aug", rainfall: 390, sst: 28.5, ndvi: 0.68, discharge: 2800, anomalies: 8 },
        { month: "Sep", rainfall: 210, sst: 28.9, ndvi: 0.72, discharge: 1500, anomalies: 2 },
        { month: "Oct", rainfall: 180, sst: 29.4, ndvi: 0.65, discharge: 900, anomalies: 1 },
        { month: "Nov", rainfall: 95, sst: 28.6, ndvi: 0.58, discharge: 500, anomalies: 0 },
        { month: "Dec", rainfall: 40, sst: 27.2, ndvi: 0.51, discharge: 350, anomalies: 0 },
    ],
    epochs: Array.from({length: 50}, (_, i) => ({
        epoch: i + 1,
        loss: Math.max(0.04, 0.85 * Math.exp(-i / 8) + (Math.random() * 0.02 - 0.01))
    })),
    tsne: Array.from({length: 50}, (_, i) => {
        const risk = i < 10 ? "CRITICAL" : i < 25 ? "HIGH" : i < 40 ? "MODERATE" : "LOW";
        let x = risk === "CRITICAL" ? 70 + Math.random()*20 : risk === "HIGH" ? 40 + Math.random()*30 : risk === "LOW" ? 10 + Math.random()*20 : 30 + Math.random()*40;
        let y = risk === "CRITICAL" ? 70 + Math.random()*20 : risk === "HIGH" ? 50 + Math.random()*30 : risk === "LOW" ? 10 + Math.random()*30 : 20 + Math.random()*40;
        return { id: i, x, y, risk };
    }),
    anomalies: [
        { zone: "Chennai", date: "2023-06-14", var: "SST", obs: "31.4°C", exp: "29.1°C", dev: "+7.9%", sev: "CRITICAL" },
        { zone: "Puducherry", date: "2023-06-18", var: "RM-NPI", obs: "0.94", exp: "0.45", dev: "+108%", sev: "CRITICAL" },
        { zone: "Kolkata", date: "2023-07-22", var: "Discharge", obs: "4100 m³/s", exp: "3100 m³/s", dev: "+32%", sev: "HIGH" },
        { zone: "Mumbai", date: "2023-08-05", var: "Rainfall", obs: "185 mm/d", exp: "45 mm/d", dev: "+310%", sev: "HIGH" },
        { zone: "Tuticorin", date: "2023-05-12", var: "Chlorophyll", obs: "0.01 mg/m³", exp: "0.4 mg/m³", dev: "-97%", sev: "MODERATE" },
        { zone: "Kochi", date: "2023-09-01", var: "SST", obs: "30.1°C", exp: "28.5°C", dev: "+5.6%", sev: "MODERATE" },
    ],
    biodiversity: Array.from({length: 20}, (_, i) => {
        const rmnpi = Math.random();
        // Negative correlation (r = -0.84)
        const bioIndex = Math.max(0.1, 1.0 - (rmnpi * 0.8) + (Math.random() * 0.2 - 0.1));
        return { id: i, rmnpi: Number(rmnpi.toFixed(2)), bioIndex: Number(bioIndex.toFixed(2)) };
    }),
    datacenter: [
        { name: 'Storage (MB)', before: 847, after: 23 },
        { name: 'Compute Cycles', before: 12400, after: 1847 },
        { name: 'Energy Time (s)', before: 340, after: 42 },
    ]
};
