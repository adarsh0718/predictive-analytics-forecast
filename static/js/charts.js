// Charts module managing ApexCharts instances for the forecasting dashboard

let forecastChart = null;
let featureImportanceChart = null;
let residualsChart = null;

/**
 * Initializes empty chart templates
 */
export function initCharts() {
    const forecastOptions = getForecastChartOptions([], [], [], [], []);
    forecastChart = new ApexCharts(document.querySelector("#chart-main-forecast"), forecastOptions);
    forecastChart.render();

    const importanceOptions = getImportanceChartOptions([], []);
    featureImportanceChart = new ApexCharts(document.querySelector("#chart-feature-importance"), importanceOptions);
    featureImportanceChart.render();

    const residualsOptions = getResidualsChartOptions([], []);
    residualsChart = new ApexCharts(document.querySelector("#chart-residuals-dist"), residualsOptions);
    residualsChart.render();
}

/**
 * Updates all charts with new API response data
 * @param {Object} data - response payload from /api/forecast
 */
export function updateCharts(data) {
    // 1. Process Forecast Data Series
    const histData = data.historical_data || [];
    const valData = data.validation_data || [];
    const fcData = data.forecast_data || [];
    
    // Combine timelines chronologically
    const allPoints = [];
    histData.forEach(p => {
        allPoints.push({ date: p.date, actual: p.actual, val: null, fc: null, lower: null, upper: null });
    });
    
    // Validation points overlap (we match dates or append)
    valData.forEach(p => {
        const existing = allPoints.find(ap => ap.date === p.date);
        if (existing) {
            existing.val = p.predicted;
        } else {
            allPoints.push({ date: p.date, actual: p.actual, val: p.predicted, fc: null, lower: null, upper: null });
        }
    });

    // Forecast points start immediately after history/validation ends
    if (fcData.length > 0 && allPoints.length > 0) {
        // Connect the last historical point to the forecast line for visual continuity
        const lastPt = allPoints[allPoints.length - 1];
        allPoints.push({
            date: lastPt.date,
            actual: null,
            val: null,
            fc: lastPt.actual || lastPt.val,
            lower: lastPt.actual || lastPt.val,
            upper: lastPt.actual || lastPt.val
        });
    }

    fcData.forEach(p => {
        allPoints.push({
            date: p.date,
            actual: null,
            val: null,
            fc: p.prediction,
            lower: p.lower_bound,
            upper: p.upper_bound
        });
    });

    // Extract series lists
    const categories = allPoints.map(p => p.date);
    const actualSeries = allPoints.map(p => p.actual);
    const valSeries = allPoints.map(p => p.val);
    const fcSeries = allPoints.map(p => p.fc);
    const lowerSeries = allPoints.map(p => p.lower);
    const upperSeries = allPoints.map(p => p.upper);

    forecastChart.updateOptions(getForecastChartOptions(categories, actualSeries, valSeries, fcSeries, lowerSeries, upperSeries));

    // 2. Process Feature Importances
    const importances = data.feature_importance || {};
    const sortedFeatures = Object.entries(importances).sort((a, b) => b[1] - a[1]);
    const featLabels = sortedFeatures.map(item => item[0]);
    const featValues = sortedFeatures.map(item => parseFloat((item[1] * 100).toFixed(1))); // Show as %

    featureImportanceChart.updateOptions(getImportanceChartOptions(featLabels, featValues));

    // 3. Process Residuals Histogram
    const residuals = data.residuals || [];
    const binned = binResiduals(residuals, 15);
    residualsChart.updateOptions(getResidualsChartOptions(binned.categories, binned.data));
}

/**
 * Custom math helper to bin residuals into histogram counts
 */
function binResiduals(residuals, numBins = 15) {
    if (!residuals || residuals.length === 0) return { categories: [], data: [] };
    const min = Math.min(...residuals);
    const max = Math.max(...residuals);
    const binWidth = (max - min) / numBins;
    
    const bins = Array(numBins).fill(0);
    const categories = [];
    
    for (let i = 0; i < numBins; i++) {
        const binStart = min + i * binWidth;
        const binEnd = binStart + binWidth;
        // Label using standard shorthand currency formatting
        categories.push(`${formatShortINR(binStart + binWidth / 2)}`);
    }
    
    residuals.forEach(val => {
        let binIdx = Math.floor((val - min) / binWidth);
        if (binIdx >= numBins) binIdx = numBins - 1;
        if (binIdx < 0) binIdx = 0;
        bins[binIdx]++;
    });
    
    return { categories, data: bins };
}

/**
 * Currency short formatter for chart labels (e.g. -₹5K)
 */
function formatShortINR(val) {
    const isNegative = val < 0;
    const absVal = Math.abs(val);
    let str = "";
    
    if (absVal >= 100000) {
        str = `₹${(absVal / 100000).toFixed(1)}L`;
    } else if (absVal >= 1000) {
        str = `₹${(absVal / 1000).toFixed(1)}K`;
    } else {
        str = `₹${Math.round(absVal)}`;
    }
    
    return isNegative ? `-${str}` : str;
}

/* ==========================================
   ApexCharts Config Generators
   ========================================== */

function getForecastChartOptions(categories, actuals, val, fc, lower, upper) {
    return {
        chart: {
            type: 'line',
            height: 350,
            background: 'transparent',
            foreColor: '#94a3b8',
            toolbar: { show: true },
            zoom: { enabled: true }
        },
        theme: { mode: 'dark' },
        colors: ['#22d3ee', '#fbbf24', '#818cf8', 'rgba(129, 140, 248, 0.25)', 'rgba(129, 140, 248, 0.25)'],
        stroke: {
            width: [3, 2.5, 3, 1, 1],
            curve: 'smooth',
            dashArray: [0, 0, 5, 4, 4]
        },
        series: [
            { name: 'Actual Sales', data: actuals },
            { name: 'Model Validation Fit', data: val },
            { name: 'Model Future Forecast', data: fc },
            { name: 'Lower Bound (95% CI)', data: lower },
            { name: 'Upper Bound (95% CI)', data: upper }
        ],
        xaxis: {
            categories: categories,
            type: 'datetime',
            labels: { datetimeUTC: false }
        },
        yaxis: {
            labels: {
                formatter: function (value) {
                    if (value === null) return "";
                    return "₹" + value.toLocaleString('en-IN');
                }
            }
        },
        tooltip: {
            shared: true,
            intersect: false,
            x: { format: 'dd MMM yyyy' },
            y: {
                formatter: function (value) {
                    if (value === undefined || value === null) return "";
                    return "₹" + value.toLocaleString('en-IN');
                }
            }
        },
        grid: {
            borderColor: 'rgba(255,255,255,0.04)',
            strokeDashArray: 3
        },
        legend: { show: false }
    };
}

function getImportanceChartOptions(labels, data) {
    return {
        chart: {
            type: 'bar',
            height: 320,
            background: 'transparent',
            foreColor: '#94a3b8',
            toolbar: { show: false }
        },
        plotOptions: {
            bar: {
                horizontal: true,
                barHeight: '60%',
                borderRadius: 4
            }
        },
        colors: ['#22d3ee'],
        series: [{ name: 'Predictive Weight (%)', data: data }],
        xaxis: {
            categories: labels,
            labels: {
                formatter: function(val) {
                    return val + "%";
                }
            }
        },
        grid: {
            borderColor: 'rgba(255,255,255,0.04)'
        },
        tooltip: {
            y: {
                formatter: function(val) {
                    return val + "%";
                }
            }
        }
    };
}

function getResidualsChartOptions(categories, data) {
    return {
        chart: {
            type: 'bar',
            height: 320,
            background: 'transparent',
            foreColor: '#94a3b8',
            toolbar: { show: false }
        },
        plotOptions: {
            bar: {
                horizontal: false,
                columnWidth: '70%',
                borderRadius: 3
            }
        },
        colors: ['#fbbf24'],
        series: [{ name: 'Error Frequency (Days)', data: data }],
        xaxis: {
            categories: categories,
            labels: {
                rotate: -45,
                style: { fontSize: '9px' }
            }
        },
        grid: {
            borderColor: 'rgba(255,255,255,0.04)'
        },
        tooltip: {
            y: {
                formatter: function(val) {
                    return val + " Days";
                }
            }
        }
    };
}
