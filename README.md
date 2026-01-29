# 🛒 BigMart Sales Prediction System

An advanced machine learning-powered web application for predicting retail sales using Random Forest regression. Built with Flask, featuring an interactive modern UI and comprehensive API endpoints.

![BigMart Sales Prediction](https://img.shields.io/badge/ML-Sales%20Prediction-blue)
![Flask](https://img.shields.io/badge/Flask-2.3.2-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

## 🌟 Features

- **Interactive Web Interface**: Modern, responsive design with real-time validation
- **Machine Learning Predictions**: Uses Random Forest algorithm for accurate sales forecasting
- **Input Validation**: Comprehensive validation with helpful error messages
- **Save Results**: Option to save prediction results to a text file
- **Mobile-Friendly**: Responsive design that works on all devices
- **Real-time Feedback**: Loading states and form validation
- **Professional UI**: Modern glassmorphism design with smooth animations

## 🚀 Quick Start

### Method 1: Using the Batch File (Windows)

```bash

run_app.bat
```

### Method 2: Manual Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
python app.py
```

### Method 3: Using Python directly

```bash
"C:/Program Files/Python311/python.exe" -m pip install -r requirements.txt
"C:/Program Files/Python311/python.exe" app.py
```

## 🌐 Accessing the Application

Once started, open your web browser and navigate to:

```
http://localhost:5000
```

## 📋 Input Fields

### Product Information

- **Item Weight**: Weight of the product in grams
- **Fat Content**: Low Fat or Regular
- **Item Visibility**: Visibility percentage (0-1)
- **Item MRP**: Maximum Retail Price in Rupees
- **Item Category**: Product category (Dairy, Snack Foods, etc.)

### Outlet Information

- **Outlet ID**: Unique outlet identifier
- **Outlet Size**: Small, Medium, or High
- **Location Type**: Tier 1 (Metro), Tier 2 (Major Cities), Tier 3 (Small Cities)
- **Outlet Type**: Type of retail outlet

## 🔧 Technical Details

### Tech Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **Machine Learning**: scikit-learn (Random Forest)
- **Data Processing**: pandas, numpy
- **UI Framework**: Custom CSS with modern design principles

### Project Structure

```
sales_prediction/
├── app.py                 # Main Flask application
├── model_rf.pkl          # Trained Random Forest model
├── requirements.txt      # Python dependencies
├── run_app.bat          # Windows batch file to start app
├── test_app.py          # Test script for validation
├── templates/
│   └── index.html       # Main web interface
└── static/
    └── style.css        # Modern CSS styling
```

### Model Information

- **Algorithm**: Random Forest Regressor
- **Features**: 9 input features covering product and outlet characteristics
- **Output**: Predicted sales value in Indian Rupees (₹)

## 🧪 Testing

Run the test script to verify everything is working:

```bash
python test_app.py
```

## 💾 Saving Results

Check the "Save result to file" option to automatically save prediction results to `prediction_results.txt` with timestamp and all input parameters.

## 🐛 Troubleshooting

### Common Issues

1. **Model not found error**

   - Ensure `model_rf.pkl` exists in the project directory
   - Check file permissions

2. **Import errors**

   - Run: `pip install -r requirements.txt`
   - Verify Python version compatibility (3.7+)

3. **Port already in use**

   - The app runs on port 5000 by default
   - Check if another application is using port 5000
   - Kill the process: `taskkill /f /im python.exe` (Windows)

4. **Template not found**
   - Ensure the `templates/` and `static/` folders exist
   - Check file paths are correct

### Health Check

Visit `http://localhost:5000/health` to check application status.

## 🎨 UI Features

- **Modern Design**: Glassmorphism and gradient backgrounds
- **Animations**: Smooth CSS animations and transitions
- **Icons**: Font Awesome icons for better visual appeal
- **Responsive**: Mobile-first design approach
- **Validation**: Real-time form validation with visual feedback
- **Loading States**: Button states during prediction processing

## 📱 Mobile Support

The application is fully responsive and works seamlessly on:

- Desktop computers
- Tablets
- Mobile phones
- All modern web browsers

## 🔮 Future Enhancements

- [ ] Add data visualization charts
- [ ] Implement user authentication
- [ ] Add batch prediction capabilities
- [ ] Include model performance metrics
- [ ] Add export functionality (PDF, Excel)
- [ ] Implement API endpoints for integration

## 📄 License

This project is for educational and demonstration purposes.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

---

**Happy Predicting! 🎯**
