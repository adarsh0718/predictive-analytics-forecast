// Data table manager for rendering historical CSV records

let tableState = {
    currentPage: 1,
    pageSize: 10,
    sortColumn: 'date',
    sortOrder: 'desc', // Default to latest dates first
    searchQuery: ''
};

let rawTableData = [];

/**
 * Loads data list and sets listeners
 */
export function initTable(data) {
    rawTableData = data || [];
    tableState.currentPage = 1;
    
    initTableListeners();
    renderTable();
}

/**
 * Registers events for searching and pagination
 */
function initTableListeners() {
    const searchInput = document.querySelector('#table-search-input');
    const btnPrev = document.querySelector('#btn-prev-page');
    const btnNext = document.querySelector('#btn-next-page');
    
    // Search input
    if (searchInput) {
        // Clear previous event listener by cloning or direct replacement
        const newSearch = searchInput.cloneNode(true);
        searchInput.parentNode.replaceChild(newSearch, searchInput);
        
        let debounceTimer = null;
        newSearch.addEventListener('input', (e) => {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
                tableState.searchQuery = e.target.value.toLowerCase().trim();
                tableState.currentPage = 1;
                renderTable();
            }, 200);
        });
    }

    // Pagination
    if (btnPrev) {
        const newPrev = btnPrev.cloneNode(true);
        btnPrev.parentNode.replaceChild(newPrev, btnPrev);
        newPrev.addEventListener('click', () => {
            if (tableState.currentPage > 1) {
                tableState.currentPage--;
                renderTable();
            }
        });
    }

    if (btnNext) {
        const newNext = btnNext.cloneNode(true);
        btnNext.parentNode.replaceChild(newNext, btnNext);
        newNext.addEventListener('click', () => {
            const processed = getProcessedData();
            const maxPage = Math.ceil(processed.length / tableState.pageSize);
            if (tableState.currentPage < maxPage) {
                tableState.currentPage++;
                renderTable();
            }
        });
    }

    // Sorting headers
    document.querySelectorAll('#records-table th').forEach(th => {
        const newTh = th.cloneNode(true);
        th.parentNode.replaceChild(newTh, th);
        
        newTh.addEventListener('click', () => {
            const col = newTh.dataset.column;
            if (!col) return;
            
            if (tableState.sortColumn === col) {
                tableState.sortOrder = tableState.sortOrder === 'asc' ? 'desc' : 'asc';
            } else {
                tableState.sortColumn = col;
                tableState.sortOrder = 'asc';
            }
            
            // Adjust arrows
            document.querySelectorAll('#records-table th').forEach(header => {
                header.classList.remove('sort-asc', 'sort-desc');
            });
            newTh.classList.add(tableState.sortOrder === 'asc' ? 'sort-asc' : 'sort-desc');
            
            renderTable();
        });
    });
}

/**
 * Filter, sort, and slice list items
 */
function getProcessedData() {
    let result = [...rawTableData];

    // Filter
    if (tableState.searchQuery !== '') {
        result = result.filter(item => {
            const dateStr = String(item.date || '').toLowerCase();
            const valStr = String(item.actual || '').toLowerCase();
            return dateStr.includes(tableState.searchQuery) || valStr.includes(tableState.searchQuery);
        });
    }

    // Sort
    const col = tableState.sortColumn;
    const isAsc = tableState.sortOrder === 'asc' ? 1 : -1;

    result.sort((a, b) => {
        let valA, valB;
        
        if (col === 'date') {
            valA = new Date(a.date);
            valB = new Date(b.date);
            return (valA - valB) * isAsc;
        } else if (col === 'actual') {
            valA = a.actual || 0;
            valB = b.actual || 0;
        } else {
            // Other features (e.g. from original CSV headers, check key matches)
            valA = a[col] || 0;
            valB = b[col] || 0;
        }

        return (valA - valB) * isAsc;
    });

    return result;
}

/**
 * Re-renders rows
 */
function renderTable() {
    const tableBody = document.querySelector('#table-body');
    const tableInfo = document.querySelector('#table-info');
    const btnPrev = document.querySelector('#btn-prev-page');
    const btnNext = document.querySelector('#btn-next-page');

    if (!tableBody) return;
    tableBody.innerHTML = '';

    const processed = getProcessedData();
    const totalCount = processed.length;
    
    // Pagination slicing
    const startIndex = (tableState.currentPage - 1) * tableState.pageSize;
    const endIndex = Math.min(startIndex + tableState.pageSize, totalCount);
    const pageData = processed.slice(startIndex, endIndex);

    // Update labels
    if (tableInfo) {
        if (totalCount === 0) {
            tableInfo.textContent = "Showing 0 of 0 entries";
        } else {
            tableInfo.textContent = `Showing ${startIndex + 1} to ${endIndex} of ${totalCount} entries`;
        }
    }

    if (btnPrev) btnPrev.disabled = tableState.currentPage === 1;
    if (btnNext) btnNext.disabled = endIndex >= totalCount || totalCount === 0;

    if (pageData.length === 0) {
        tableBody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-muted); padding: 2rem;">No historical records found.</td></tr>`;
        return;
    }

    pageData.forEach(row => {
        const tr = document.createElement('tr');
        
        const salesVal = `₹${Math.round(row.actual).toLocaleString('en-IN')}`;
        
        // Match optional columns from base raw properties if present (simulated db data has these)
        // Marketing Spend, Store Traffic, Discount %, Holiday?
        const marketing = row.marketing !== undefined ? `₹${Math.round(row.marketing).toLocaleString('en-IN')}` : 'N/A';
        const traffic = row.traffic !== undefined ? row.traffic.toLocaleString() : 'N/A';
        const discount = row.discount !== undefined ? `${row.discount}%` : 'N/A';
        const holiday = row.holiday !== undefined ? (row.holiday === 1 ? 'Yes' : 'No') : 'N/A';
        
        tr.innerHTML = `
            <td>${row.date}</td>
            <td style="text-align: right; font-weight: 500; color: var(--text-primary);">${salesVal}</td>
            <td style="text-align: right;">${marketing}</td>
            <td style="text-align: right;">${traffic}</td>
            <td style="text-align: right; color: var(--accent-amber);">${discount}</td>
            <td style="text-align: right; color: ${holiday === 'Yes' ? 'var(--accent-emerald)' : 'var(--text-muted)'}">${holiday}</td>
        `;
        tableBody.appendChild(tr);
    });
}
