import streamlit as st


class About:
    def aboutsection(self):
        try:
            st.markdown('<div style="text-align: center"><h2> Our AutoML Adventure 🚀 </div>', unsafe_allow_html=True)
            st.markdown(
            '<div style="background-color: #3498db; padding: 20px; border-radius: 10px; box-shadow: 5px 5px 5px #888888;">'
            '<h1 style="color: white; text-align: center;">Welcome to Auto Machine Learning App</h1>'
            '<p style="color: white; text-align: center; font-size: 18px;">Your Gateway to Automated Machine Learning.</p>'
            '</div>',
            unsafe_allow_html=True
            )

            st.write("")
            st.markdown('<div style="text-align: center"><h3> 🌟 Description </div>', unsafe_allow_html=True)
            st.write("Auto Machine Learning is the ultimate solution for automating your machine learning tasks. It empowers you to effortlessly analyze and interpret your data, whether you're tackling classification problems or regression challenges. With our intuitive interface, you can streamline data preprocessing, model selection, and results visualization.")
            
            st.write("🌟 Once upon a time, in the world of data, we embarked on an incredible journey as freshers, armed with determination and passion for machine learning.")
        
            st.write("🛠️ We realized that data analysis can be challenging, especially when you're just starting out. And that's when the idea of Auto Machine Learning was born.")

            st.write("🚀 Our goal was to create a tool that would make data-driven insights accessible to everyone, regardless of their experience level. We wanted to automate the machine learning process to make it effortless and user-friendly.")
        
            st.write("💡 With continuous learning, late-night coding, and endless cups of coffee, we developed an intuitive interface that automates data preprocessing, model selection, and result visualization.")
        
            st.write("📊 Auto Machine Learning is not just a project; it's our passion. It's our way of saying, 'You don't need to be an expert to analyze data effectively.'")

            st.write("📧 If you have any questions, feedback, or just want to say 'hi', don't hesitate to reach out. We're here to support you on your data journey!")
            st.write("")
        
            st.markdown('<div style="text-align: center"><h3> 📧 Contact Information: </div>', unsafe_allow_html=True)
            st.write("If you have any questions or feedback, feel free to contact us:")
            st.write("Email: prvnbharti@gmail.com")
            st.write("Email: ketkishinde2904@gmail.com")
            st.write("")
            
            st.markdown('<div style="text-align: center"><h3> 💖 Spread the Love: Share Your Feedback 💬 </div>', unsafe_allow_html=True)
            
            # "Like" button
            if st.button("👍 Like This Project"):
                st.write("Thank you for your support! 😊")

        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")


    




     