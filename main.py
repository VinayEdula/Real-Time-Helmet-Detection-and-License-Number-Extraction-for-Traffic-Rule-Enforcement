from ultralytics import YOLO
import cv2
import pytesseract
import os
import sqlite3

# Function to create the table if it doesn't exist
def create_table():
    # Connect to the SQLite database
    conn = sqlite3.connect("vehicle_data.db")
    # Create a cursor object to execute SQL statements
    c = conn.cursor()
    # Create the "vehicles" table with two columns: "vehicle_number" and "bike_image_path"
    # The "IF NOT EXISTS" clause ensures that the table is only created if it doesn't already exist
    c.execute("CREATE TABLE IF NOT EXISTS vehicles (vehicle_number TEXT, bike_image_path TEXT)")
    # Commit the changes to the database
    conn.commit()
    # Close the database connection
    conn.close()

# Function to insert a record into the "vehicles" table
def insert_record(vehicle_number, bike_image_path):
    # Connect to the SQLite database
    conn = sqlite3.connect("vehicle_data.db")
    # Create a cursor object to execute SQL statements
    c = conn.cursor()
    # Insert the record into the "vehicles" table using parameterized SQL statement
    c.execute("INSERT INTO vehicles VALUES (?, ?)", (vehicle_number, bike_image_path))
    # Commit the changes to the database
    conn.commit()
    # Close the database connection
    conn.close()


# Initialize YOLO models
person_bike_model = YOLO(r"C:\Users\Vinay Edula\Downloads\Person-Bike Results\best.pt")
helmet_model = YOLO(r"C:\Users\Vinay Edula\Downloads\Bike Helmets results\best.pt")
number_plate_model = YOLO(r"C:\Users\Vinay Edula\Downloads\Vehicle number results\best.pt")

# Set up Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"path/to/tesseract_executable"  # Update with the path to your Tesseract OCR executable

output_dir = r"C:\Users\Vinay Edula\Downloads\out"  # Directory to save the output images
# Set up video capture
video_capture = cv2.VideoCapture(0)  # Use 0 for the default camera or provide the desired camera index

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    # Process frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Detect person on a bike
    person_bike_results = person_bike_model.predict(img)

    # Process each detection result
    for r in person_bike_results:
        boxes = r.boxes
        # Filter detections for person on a bike
        for box in boxes:
            cls = box.cls
            print(person_bike_model.names[int(cls)], person_bike_model.names[int(cls)] == "Person_Bikes")
            if person_bike_model.names[int(cls)] == "Person_Bike":
                # Crop person on a bike image
                x1, y1, x2, y2 = box.xyxy[0]
                person_bike_image = frame[int(y1):int(y2), int(x1):int(x2)]

                # Detect helmet on the person
                helmet_results = helmet_model.predict(person_bike_image)

                # Process each helmet detection result
                for hr in helmet_results:
                    h_boxes = hr.boxes
                    # Filter detections for no helmet
                    for h_bo in h_boxes:
                        h_cls = h_bo.cls
                        if not helmet_model.names[int(h_cls)] == "With Helmet" :
                            # Extract number plate from the person bike image
                            number_plate_results = number_plate_model.predict(person_bike_image)

                            # Process each number plate detection result
                            for npr in number_plate_results:
                                np_boxes = npr.boxes
                                # Filter detections for number plate
                                for np_box in np_boxes:
                                    np_cls = np_box.cls
                                    print(number_plate_model.names[int(np_cls)])
                                    if number_plate_model.names[int(np_cls)] == "number_plate":
                                        # Crop number plate image
                                        np_x1, np_y1, np_x2, np_y2 = np_box.xyxy[0]
                                        number_plate_image = person_bike_image[int(np_y1):int(np_y2),
                                                             int(np_x1):int(np_x2)]
                                        # Save the cropped number plate image
                                        output_file = f"person_violation_{image_file}"
                                        output_path = os.path.join(output_dir, output_file)
                                        cv2.imwrite(output_path, person_bike_image)

                                        # Perform OCR on the number plate image
                                        gray = cv2.cvtColor(number_plate_image, cv2.COLOR_BGR2GRAY)
                                        text = pytesseract.image_to_string(gray)
                                        # Example usage
                                        # Create the "vehicles" table if it doesn't exist
                                        create_table()
                                        # Insert two records into the "vehicles" table
                                        insert_record(text, output_path)
                                        # Print the extracted text
                                        print("Number Plate Text:", text)
