import json
import pandas as pd

class Parser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.stroke_types = []
    
    def load_data(self):
        """Load JSON data from file"""
        try:
            with open(self.file_path, 'r') as file:
                self.data = json.load(file)
            return True
        except FileNotFoundError:
            print(f"Error: File '{self.file_path}' not found.")
            return False
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in '{self.file_path}'.")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False
    
    def print_basic_info(self):
        """Print basic match information"""
        if not self.data:
            print("No data loaded. Call load_data() first.")
            return
        
        print("=== Basic Match Information ===")
        print(self.data)
        # Uncomment if you want to see StrokeType directly
        # print("\nStroke Types:", self.data.get("StrokeType", "Not found"))
    
    def analyze_points(self):
        """Analyze and print point-by-point information"""
        if not self.data:
            print("No data loaded. Call load_data() first.")
            return
        
        print("\n=== Point Analysis ===")
        for point_key, point_value in self.data.items():
            print(f"\nPoint: {point_key}")
            
            try:
                video_info = point_value["VideoInfo"]
                point_info = point_value["PointInfo"]
                
                print(f"Visibility: {video_info['Visibility']}")
                print(f"Point Duration: {point_info['PointEnd'] - point_info['PointBegin']}")
                print(f"Win Cause: {point_info['WonEnd']['Cause']}")
            except KeyError as e:
                print(f"Missing expected data in point: {e}")
    
    def extract_stroke_types(self):
        """Extract all stroke types from the data"""
        if not self.data:
            print("No data loaded. Call load_data() first.")
            return
        
        self.stroke_types = []
        
        def traverse_json(data):
            if isinstance(data, dict):
                for key, value in data.items():
                    if key == "StrokeType":
                        self.stroke_types.append(value)
                    if isinstance(value, (dict, list)):
                        traverse_json(value)
            elif isinstance(data, list):
                for item in data:
                    traverse_json(item)
        
        traverse_json(self.data)
        return self.stroke_types
    
    def save_stroke_types_to_csv(self, output_file="stroke_types.csv"):
        """Save extracted stroke types to CSV"""
        if not self.stroke_types:
            self.extract_stroke_types()
        
        df = pd.DataFrame({"StrokeType": self.stroke_types})
        df.to_csv(output_file, index=False)
        print(f"Stroke types saved to {output_file}")
    
    def save_processed_data(self, output_file="output.json"):
        """Save processed data back to JSON"""
        if not self.data:
            print("No data loaded. Call load_data() first.")
            return
        
        with open(output_file, 'w') as file:
            json.dump(self.data, file, indent=4)
        print(f"Processed data saved to {output_file}")


# Example usage:
if __name__ == "__main__":
    parser1 = Parser(
        'json/2018-27-2R16-MS-Anthony Sinisuka GINTING (INA)-Kento MOMOTA (JPN)-BLIBLI Indonesia Open-1080p.json'
    )
    parser2 = Parser(
        'json\2018-38-5F-MS-Anthony Sinisuka GINTING (INA)-Kento MOMOTA (JPN)-Victor China Open-1080p.json'
    )
    parser3 = Parser(
        'json\2019-04-3QF-MS-Anthony Sinisuka GINTING (INA)-Kento MOMOTA (JPN)-Daihatsu Indonesia Masters-1080p.json'
    )
    parser4 = Parser(
        'json\2019-15-5F-MS-Anthony Sinisuka GINTING (INA)-Kento MOMOTA (JPN)-Singapore Open 2019-1080p.json'
    )
    parser5 = Parser(
        'json\2019-20-6SF-MS-Anthony Sinisuka GINTING (INA)-Kento MOMOTA (JPN)-Sudirman Cup-1080p.json'
    )
    parser6 = Parser(
        'json/2018-27-2R16-MS-Anthony Sinisuka GINTING (INA)-Kento MOMOTA (JPN)-BLIBLI Indonesia Open-1080p.json'
    )
    parser7 = Parser(
        'json\2019-38-5F-MS-Anthony Sinisuka GINTING (INA)-Kento MOMOTA (JPN)-Victor China Open-1080p.json'
    )
    parser8 = Parser(
        'json\2019-43-3QF-MS-Anthony Sinisuka GINTING (INA)-Kento Momota (JPN)-YONEX French Open-1080p-50fps.json'
    )
    parser9 = Parser(
        'json\2019-50-5F-MS-Anthony Sinisuka GINTING (INA)-Kento MOMOTA (JPN)-HSBC BWF World Tour Finals-1080p.json'
    )
    
    # Load and process the data
    if parser1.load_data():
        parser1.print_basic_info()
        parser1.analyze_points()
        parser1.save_stroke_types_to_csv()
        parser1.save_processed_data()

    if parser2.load_data():
        parser2.print_basic_info()
        parser2.analyze_points()
        parser2.save_stroke_types_to_csv()
        parser2.save_processed_data()
    
    if parser3.load_data():
        parser3.print_basic_info()
        parser3.analyze_points()
        parser3.save_stroke_types_to_csv()
        parser3.save_processed_data()
    
    if parser4.load_data():
        parser4.print_basic_info()
        parser4.analyze_points()
        parser4.save_stroke_types_to_csv()
        parser4.save_processed_data()
    
    if parser5.load_data():
        parser5.print_basic_info()
        parser5.analyze_points()
        parser5.save_stroke_types_to_csv()
        parser5.save_processed_data()
    
    if parser6.load_data():
        parser6.print_basic_info()
        parser6.analyze_points()
        parser6.save_stroke_types_to_csv()
        parser6.save_processed_data()

    if parser7.load_data():
        parser7.print_basic_info()
        parser7.analyze_points()
        parser7.save_stroke_types_to_csv()
        parser7.save_processed_data()

    if parser8.load_data():
        parser8.print_basic_info()
        parser8.analyze_points()
        parser8.save_stroke_types_to_csv()
        parser8.save_processed_data()

    if parser9.load_data():
        parser9.print_basic_info()
        parser9.analyze_points()
        parser9.save_stroke_types_to_csv()
        parser9.save_processed_data()