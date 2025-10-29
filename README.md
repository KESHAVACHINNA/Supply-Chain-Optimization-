# Supply-Chain-Optimization

This project is a Streamlit web application built to optimize supply chain logistics using Linear Programming and Machine Learning techniques.
It helps minimize total transportation and storage costs by simulating different warehouse–store allocations and demand scenarios.

With an interactive UI powered by Streamlit, users can modify parameters like cost, demand, and capacity in real time — making it an excellent tool for decision-making and supply chain analysis.

🔗 Live Demo: [Click here to view the app](https://wczpkmcheve2bp6n5laze3.streamlit.app/)

 Repository: GitHub Repo Link
 (Add your GitHub repo link here)

🧠 Features

🔢 Optimize warehouse–store allocations using Linear Programming (via Google OR-Tools).

📉 Minimize logistics costs and improve delivery efficiency.

⚙️ Adjustable parameters for demand, cost, and distance.

📊 Interactive results visualization with Streamlit charts and tables.

🧠 Optionally integrates ML-based demand forecasting (using XGBoost / Prophet).

🧰 Tech Stack

Programming Language: Python

Framework: Streamlit

Optimization Engine: OR-Tools (Linear Programming)

Data Handling: Pandas, NumPy

Visualization: Matplotlib / Plotly

Machine Learning (optional): scikit-learn, XGBoost, Prophet

⚙️ How It Works

Upload or input your warehouse and store data (demand, distance, capacity, cost).

The app formulates and solves an optimization model to minimize total cost.

Displays the optimal allocation of stock between warehouses and stores.

Provides visual insights and performance metrics interactively.

💻 Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/yourusername/supply-chain-optimization.git
cd supply-chain-optimization

2️⃣ Create a virtual environment
python -m venv venv
source venv/bin/activate     # for Mac/Linux
venv\Scripts\activate        # for Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the Streamlit app
streamlit run app.py


Then open your browser and go to:

http://localhost:8501

📁 Folder Structure
📦 supply-chain-optimization/
 ┣ 📜 app.py
 ┣ 📜 requirements.txt
 ┣ 📜 README.md
 ┣ 📂 data/
 ┃ ┗ dataset.csv
 ┣ 📂 models/
 ┃ ┗ trained_model.pkl
 ┗ 📂 utils/
   ┗ optimization.py

📈 Example Output
Warehouse	Store	Units	Cost
W1	S1	500	₹25,000
W2	S3	800	₹40,000

Total Optimized Cost: ₹65,000
(Replace with your own output or screenshot)

🌐 Deployment

Deployed on Streamlit Cloud
You can deploy your own version by connecting this repo to Streamlit Cloud
 or Render
.

Start Command:

streamlit run app.py --server.port 10000 --server.address 0.0.0.0

📚 Future Enhancements

Add inventory forecasting using ML models (Prophet, LSTM).

Include interactive geospatial visualization using Folium.

Support multi-objective optimization (cost, time, emissions).

Integrate Power BI or Tableau dashboards for business reporting.
