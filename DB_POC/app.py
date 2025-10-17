import streamlit as st
import json
 
st.set_page_config(page_title="Invoice Viewer", layout="centered")
 
st.title("📄 Invoice Dashboard")
st.write("Upload an invoice JSON file (output from Doc AI or simulation)")
 
uploaded_file = st.file_uploader("Upload JSON Invoice", type=["json"])
 
if uploaded_file:
    invoice_data = json.load(uploaded_file)
 
    st.subheader("🧾 Invoice Summary")
    col1, col2 = st.columns(2)
    col1.write(f"**Invoice #:** {invoice_data['invoiceNumber']}")
    col1.write(f"**Invoice Date:** {invoice_data['invoiceDate']}")
    col1.write(f"**Due Date:** {invoice_data['dueDate']}")
    col2.write(f"**PO #:** {invoice_data['poNumber']}")
    col2.write(f"**Total Amount:** ${invoice_data['totalAmountDue']:,.2f}")
 
    st.subheader("🏨 Billing & Shipping")
    st.write(f"**Bill To:** {invoice_data['billTo']['name']}, {invoice_data['billTo']['address']}")
    st.write(f"**Ship To:** {invoice_data['shipTo']['name']}, {invoice_data['shipTo']['address']}")
 
    st.subheader("📦 Line Items")
    st.table(invoice_data['lineItems'])
 
    st.subheader("💳 Payment Info")
    st.write(f"**Bank:** {invoice_data['paymentMethods']['bankTransfer']['bankName']}")
    st.write(f"**Account:** {invoice_data['paymentMethods']['bankTransfer']['accountName']}")
    st.write(f"**SWIFT/BIC:** {invoice_data['paymentMethods']['bankTransfer']['swiftCode']}")
 
    st.subheader("📝 Notes")
    st.info(invoice_data['notes'])
 
else:
    st.info("Upload a JSON file to begin.")
 