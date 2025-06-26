function updateProducts() {
    const category = document.getElementById("category").value;
    const productDropdown = document.getElementById("product");

    productDropdown.innerHTML = '<option value="">-- Select Product --</option>';

    if (productMap[category]) {
        productMap[category].forEach(product => {
            const option = document.createElement("option");
            option.value = product;
            option.textContent = product;
            productDropdown.appendChild(option);
        });
    }
}
