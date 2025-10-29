import streamlit as st
import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Supply Chain Optimizer",
    page_icon="🚚",
    layout="wide"
)

# Title and description
st.title("🚚 Supply Chain Optimization System")
st.markdown("**Minimize logistics costs and optimize delivery efficiency using Linear Programming**")

# Sidebar for inputs
st.sidebar.header("Configuration")

# Number of warehouses and customers
num_warehouses = st.sidebar.number_input("Number of Warehouses", min_value=2, max_value=10, value=3)
num_customers = st.sidebar.number_input("Number of Customers", min_value=2, max_value=15, value=5)

# Initialize session state for data
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Function to generate sample data
def generate_sample_data(n_warehouses, n_customers):
    np.random.seed(42)
    
    # Warehouse capacities
    capacities = np.random.randint(100, 500, n_warehouses)
    
    # Customer demands
    demands = np.random.randint(50, 200, n_customers)
    
    # Transportation costs (warehouse to customer)
    costs = np.random.randint(5, 50, (n_warehouses, n_customers))
    
    # Warehouse fixed costs
    fixed_costs = np.random.randint(1000, 5000, n_warehouses)
    
    return capacities, demands, costs, fixed_costs

# Generate or use custom data
if st.sidebar.button("Generate Sample Data") or not st.session_state.initialized:
    capacities, demands, costs, fixed_costs = generate_sample_data(num_warehouses, num_customers)
    st.session_state.capacities = capacities
    st.session_state.demands = demands
    st.session_state.costs = costs
    st.session_state.fixed_costs = fixed_costs
    st.session_state.initialized = True

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Input Data", "⚙️ Optimization", "📈 Results", "💡 Insights"])

with tab1:
    st.header("Supply Chain Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Warehouse Capacities")
        warehouse_df = pd.DataFrame({
            'Warehouse': [f'W{i+1}' for i in range(num_warehouses)],
            'Capacity': st.session_state.capacities,
            'Fixed Cost': st.session_state.fixed_costs
        })
        st.dataframe(warehouse_df, use_container_width=True)
        
        # Edit capacities
        with st.expander("Edit Warehouse Data"):
            for i in range(num_warehouses):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.session_state.capacities[i] = st.number_input(
                        f"W{i+1} Capacity", 
                        value=int(st.session_state.capacities[i]),
                        key=f"cap_{i}"
                    )
                with col_b:
                    st.session_state.fixed_costs[i] = st.number_input(
                        f"W{i+1} Fixed Cost", 
                        value=int(st.session_state.fixed_costs[i]),
                        key=f"fixed_{i}"
                    )
    
    with col2:
        st.subheader("Customer Demands")
        customer_df = pd.DataFrame({
            'Customer': [f'C{i+1}' for i in range(num_customers)],
            'Demand': st.session_state.demands
        })
        st.dataframe(customer_df, use_container_width=True)
        
        # Edit demands
        with st.expander("Edit Customer Demands"):
            for i in range(num_customers):
                st.session_state.demands[i] = st.number_input(
                    f"C{i+1} Demand", 
                    value=int(st.session_state.demands[i]),
                    key=f"dem_{i}"
                )
    
    st.subheader("Transportation Cost Matrix (Warehouse → Customer)")
    cost_df = pd.DataFrame(
        st.session_state.costs,
        columns=[f'C{i+1}' for i in range(num_customers)],
        index=[f'W{i+1}' for i in range(num_warehouses)]
    )
    st.dataframe(cost_df, use_container_width=True)
    
    # Heatmap of costs
    fig_cost = px.imshow(
        st.session_state.costs,
        labels=dict(x="Customer", y="Warehouse", color="Cost"),
        x=[f'C{i+1}' for i in range(num_customers)],
        y=[f'W{i+1}' for i in range(num_warehouses)],
        color_continuous_scale="RdYlGn_r",
        title="Transportation Cost Heatmap"
    )
    st.plotly_chart(fig_cost, use_container_width=True)

with tab2:
    st.header("Optimization Engine")
    
    st.markdown("""
    **Objective:** Minimize total logistics costs including:
    - Transportation costs (warehouse to customer)
    - Fixed warehouse operational costs
    
    **Constraints:**
    - Meet all customer demands
    - Respect warehouse capacities
    - Binary decisions for warehouse operations
    """)
    
    if st.button("🚀 Run Optimization", type="primary"):
        with st.spinner("Solving optimization problem..."):
            # Create solver
            solver = pywraplp.Solver.CreateSolver('SCIP')
            
            if not solver:
                st.error("Solver not available!")
            else:
                # Decision variables
                # x[i][j] = amount shipped from warehouse i to customer j
                x = {}
                for i in range(num_warehouses):
                    for j in range(num_customers):
                        x[i, j] = solver.NumVar(0, solver.infinity(), f'x_{i}_{j}')
                
                # y[i] = 1 if warehouse i is used, 0 otherwise
                y = {}
                for i in range(num_warehouses):
                    y[i] = solver.BoolVar(f'y_{i}')
                
                # Objective function
                objective = solver.Objective()
                
                # Transportation costs
                for i in range(num_warehouses):
                    for j in range(num_customers):
                        objective.SetCoefficient(x[i, j], st.session_state.costs[i][j])
                
                # Fixed costs
                for i in range(num_warehouses):
                    objective.SetCoefficient(y[i], st.session_state.fixed_costs[i])
                
                objective.SetMinimization()
                
                # Constraints
                # 1. Meet customer demands
                for j in range(num_customers):
                    constraint = solver.Constraint(st.session_state.demands[j], st.session_state.demands[j])
                    for i in range(num_warehouses):
                        constraint.SetCoefficient(x[i, j], 1)
                
                # 2. Respect warehouse capacities
                for i in range(num_warehouses):
                    constraint = solver.Constraint(0, st.session_state.capacities[i])
                    for j in range(num_customers):
                        constraint.SetCoefficient(x[i, j], 1)
                
                # 3. Link warehouse usage with shipments
                for i in range(num_warehouses):
                    for j in range(num_customers):
                        constraint = solver.Constraint(-solver.infinity(), 0)
                        constraint.SetCoefficient(x[i, j], 1)
                        constraint.SetCoefficient(y[i], -st.session_state.capacities[i])
                
                # Solve
                status = solver.Solve()
                
                if status == pywraplp.Solver.OPTIMAL:
                    st.success("✅ Optimal solution found!")
                    
                    # Store results
                    st.session_state.solution = {}
                    st.session_state.solution['total_cost'] = solver.Objective().Value()
                    st.session_state.solution['shipments'] = np.zeros((num_warehouses, num_customers))
                    st.session_state.solution['warehouse_used'] = []
                    
                    for i in range(num_warehouses):
                        if y[i].solution_value() > 0.5:
                            st.session_state.solution['warehouse_used'].append(i)
                        for j in range(num_customers):
                            st.session_state.solution['shipments'][i][j] = x[i, j].solution_value()
                    
                    # Calculate costs breakdown
                    transport_cost = sum(
                        st.session_state.costs[i][j] * x[i, j].solution_value()
                        for i in range(num_warehouses)
                        for j in range(num_customers)
                    )
                    fixed_cost = sum(
                        st.session_state.fixed_costs[i] * y[i].solution_value()
                        for i in range(num_warehouses)
                    )
                    
                    st.session_state.solution['transport_cost'] = transport_cost
                    st.session_state.solution['fixed_cost'] = fixed_cost
                    
                    st.metric("Total Optimized Cost", f"${st.session_state.solution['total_cost']:,.2f}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Transportation Cost", f"${transport_cost:,.2f}")
                    with col2:
                        st.metric("Fixed Costs", f"${fixed_cost:,.2f}")
                    with col3:
                        st.metric("Active Warehouses", len(st.session_state.solution['warehouse_used']))
                    
                else:
                    st.error("No optimal solution found!")

with tab3:
    st.header("Optimization Results")
    
    if 'solution' in st.session_state:
        # Shipment matrix
        st.subheader("Optimal Shipment Allocation")
        shipment_df = pd.DataFrame(
            st.session_state.solution['shipments'],
            columns=[f'C{i+1}' for i in range(num_customers)],
            index=[f'W{i+1}' for i in range(num_warehouses)]
        )
        st.dataframe(shipment_df.style.highlight_max(axis=0), use_container_width=True)
        
        # Shipment heatmap
        fig_shipment = px.imshow(
            st.session_state.solution['shipments'],
            labels=dict(x="Customer", y="Warehouse", color="Shipment"),
            x=[f'C{i+1}' for i in range(num_customers)],
            y=[f'W{i+1}' for i in range(num_warehouses)],
            color_continuous_scale="Blues",
            title="Optimal Shipment Allocation"
        )
        st.plotly_chart(fig_shipment, use_container_width=True)
        
        # Warehouse utilization
        st.subheader("Warehouse Utilization")
        warehouse_usage = []
        for i in range(num_warehouses):
            total_shipped = sum(st.session_state.solution['shipments'][i])
            utilization = (total_shipped / st.session_state.capacities[i]) * 100
            warehouse_usage.append({
                'Warehouse': f'W{i+1}',
                'Shipped': total_shipped,
                'Capacity': st.session_state.capacities[i],
                'Utilization (%)': utilization,
                'Status': 'Active' if i in st.session_state.solution['warehouse_used'] else 'Inactive'
            })
        
        usage_df = pd.DataFrame(warehouse_usage)
        st.dataframe(usage_df, use_container_width=True)
        
        # Utilization chart
        fig_util = go.Figure()
        fig_util.add_trace(go.Bar(
            x=usage_df['Warehouse'],
            y=usage_df['Utilization (%)'],
            marker_color=['green' if s == 'Active' else 'red' for s in usage_df['Status']],
            text=usage_df['Utilization (%)'].round(1),
            textposition='auto',
        ))
        fig_util.update_layout(
            title="Warehouse Utilization Percentage",
            xaxis_title="Warehouse",
            yaxis_title="Utilization (%)",
            showlegend=False
        )
        st.plotly_chart(fig_util, use_container_width=True)
        
        # Cost breakdown
        st.subheader("Cost Breakdown")
        cost_data = pd.DataFrame({
            'Category': ['Transportation', 'Fixed Costs', 'Total'],
            'Cost': [
                st.session_state.solution['transport_cost'],
                st.session_state.solution['fixed_cost'],
                st.session_state.solution['total_cost']
            ]
        })
        
        fig_cost_breakdown = px.pie(
            cost_data[cost_data['Category'] != 'Total'],
            values='Cost',
            names='Category',
            title='Cost Distribution',
            color_discrete_sequence=['#3498db', '#e74c3c']
        )
        st.plotly_chart(fig_cost_breakdown, use_container_width=True)
        
    else:
        st.info("👈 Run the optimization in the 'Optimization' tab to see results")

with tab4:
    st.header("Business Insights & Recommendations")
    
    if 'solution' in st.session_state:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✅ Key Findings")
            
            # Active warehouses
            active_pct = (len(st.session_state.solution['warehouse_used']) / num_warehouses) * 100
            st.write(f"- **{len(st.session_state.solution['warehouse_used'])} out of {num_warehouses}** warehouses are actively used ({active_pct:.0f}%)")
            
            # Cost efficiency
            transport_pct = (st.session_state.solution['transport_cost'] / st.session_state.solution['total_cost']) * 100
            st.write(f"- Transportation costs represent **{transport_pct:.1f}%** of total costs")
            
            # Average utilization
            avg_util = sum(
                sum(st.session_state.solution['shipments'][i]) / st.session_state.capacities[i]
                for i in st.session_state.solution['warehouse_used']
            ) / len(st.session_state.solution['warehouse_used']) * 100
            st.write(f"- Average utilization of active warehouses: **{avg_util:.1f}%**")
            
        with col2:
            st.subheader("💡 Recommendations")
            
            if active_pct < 70:
                st.write("- Consider consolidating operations to fewer warehouses")
            
            if avg_util < 60:
                st.write("- Low utilization detected - potential for capacity optimization")
            
            if transport_pct > 70:
                st.write("- High transportation costs - consider warehouse location strategy")
            
            st.write("- Monitor demand patterns for dynamic allocation adjustments")
        
        # Download results
        st.subheader("📥 Export Results")
        
        result_data = shipment_df.copy()
        csv = result_data.to_csv()
        
        st.download_button(
            label="Download Shipment Plan (CSV)",
            data=csv,
            file_name="optimal_shipment_plan.csv",
            mime="text/csv"
        )
        
    else:
        st.info("👈 Run the optimization to see insights and recommendations")

# Footer
st.markdown("---")
st.markdown("**Supply Chain Optimization System** | Powered by OR-Tools & Streamlit")
