# E-commerce Sales Dashboard

This project is an interactive sales dashboard built using Python, Streamlit, and Prophet for sales forecasting. It visualizes e-commerce sales data and provides forecasting with user-friendly interactive charts.

## Features

- Display key sales metrics: Total Sales, Total Profit, Total Orders
- Filter sales data by category and sub-category
- Visualize profit by region
- Monthly sales trend visualization with date filtering
- Sales forecasting using Facebook Prophet
- Interactive plots with zoom, pan, and download options

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/sravyamangipudi11/E-commerce-dashboard.git
   cd E-commerce-dashboard
Create and activate a virtual environment (recommended):

bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
Install the required packages:

bash
Copy
Edit
pip install -r requirements.txt
Usage
Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py
This will open the dashboard in your default browser.

Data
The dataset used is Superstore.csv, containing sales data for the e-commerce store.

Dependencies
pandas

numpy

matplotlib

plotly

prophet

streamlit

License
This project is licensed under the MIT License.