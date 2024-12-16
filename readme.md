To run the given Flask application on your local machine, follow these steps:

---

### **Prerequisites**
1. **Python Installed**: Ensure Python 3.7+ is installed. Check by running:
   ```bash
   python --version
   ```
   or
   ```bash
   python3 --version
   ```

2. **Install Required Packages**:
   - `Flask`
   - `Pillow` (for image handling)
   - Any SDKs or dependencies, such as `inference_sdk` (ensure this is properly installed).

   Use `pip` to install the dependencies:
   ```bash
   pip install flask pillow
   ```

   If `inference_sdk` is a custom or external library, install it using its provided instructions or by running:
   ```bash
   pip install inference-sdk
   ```

3. **Check for API Key**: Ensure your `InferenceHTTPClient` API key is valid and the API endpoint (`https://detect.roboflow.com`) is accessible.

---

### **Steps to Run the Code**
1. **Save the Code**:
   Save your Python code to a file, e.g., `app.py`.

2. **Run the Flask App**:
   Open a terminal, navigate to the folder containing the file, and run:
   ```bash
   python app.py
   ```

   If your system uses `python3` instead:
   ```bash
   python3 app.py
   ```

3. **Access the API**:
   - The app will run on `http://127.0.0.1:5000` (or `http://localhost:5000`).
   - The `/process-image` endpoint is available for handling POST requests.

---

### **Testing the Endpoint**
1. **Using `curl` (Command Line)**:
   ```bash
   curl -X POST -F "image=@<path_to_xray_image>" http://127.0.0.1:5000/process-image
   ```
   Replace `<path_to_xray_image>` with the path to an X-ray image file on your machine.

2. **Using Postman**:
   - Open Postman.
   - Set the method to `POST`.
   - URL: `http://127.0.0.1:5000/process-image`.
   - In the **Body** tab, select `form-data`.
   - Add a key named `image`, set the type to `File`, and upload an image.
   - Send the request.
   
   ***Example***
    ![s4](https://github.com/user-attachments/assets/7ba22fd2-66df-4af5-9d4c-70739dc8f5f7)

3. **Using a Python Script**:
   ```python
   import requests

   url = "http://127.0.0.1:5000/process-image"
   files = {'image': open('<path_to_xray_image>', 'rb')}  # Replace with actual file path
   response = requests.post(url, files=files)
   print(response.json())
   ```
