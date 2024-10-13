import streamlit as st

# Main function
def main():
    # Set page config
    st.set_page_config(page_title="Dance Monkey", layout="centered")
    
    # Start Page
    if 'page' not in st.session_state:
        st.session_state.page = 'start'

    # Start Page
    if st.session_state.page == 'start':
        # Set the background image
        st.markdown(
            """
            <style>
            .reportview-container {
                background: url("first.jpg");
                background-size: cover;
                height: 100vh;
                color: white;
            }
            .main {
                text-align: center;
            }
            </style>
            """, unsafe_allow_html=True
        )
        
        st.markdown("<h1 class='main'>Welcome to Dance Monkey</h1>", unsafe_allow_html=True)
        st.markdown("<h2 class='main'>Let's Dance!</h2>", unsafe_allow_html=True)
        
        if st.button("Start", key="start_button"):
            st.session_state.page = 'selection'

    # Second Page with buttons
    elif st.session_state.page == 'selection':
        st.title("Choose Your Dance")
        st.markdown("<h2 style='text-align: center;'>Select a Dance Style:</h2>", unsafe_allow_html=True)

        dance_options = ["Two Step", "Moon Walk", "Shoulder Lean", "Salsa", "Grapevine"]
        
        for dance in dance_options:
            if st.button(dance):
                st.session_state.page = dance.lower().replace(" ", "_")

    # Import dance form pages based on selection
    elif st.session_state.page == 'two_step':
        import two_step  # Ensure you have two_step.py created
        two_step.main()

    elif st.session_state.page == 'moon_walk':
        import moon_walk  # Ensure you have moon_walk.py created
        moon_walk.main()

    elif st.session_state.page == 'shoulder_lean':
        import shoulder_lean  # Ensure you have shoulder_lean.py created
        shoulder_lean.main()

    elif st.session_state.page == 'salsa':
        import salsa  # Ensure you have salsa.py created
        salsa.main()

    elif st.session_state.page == 'grapevine':
        import grapevine  # Ensure you have grapevine.py created
        grapevine.main()

# Run the app
if __name__ == "__main__":
    main()
