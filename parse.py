import json
import pandas as pd

class parse:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.last_stroke_types = []  # Changed to store last stroke types
    
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
    
    def extract_last_stroke_types(self):
        """Extract the last stroke type from each point"""
        if not self.data:
            print("No data loaded. Call load_data() first.")
            return
        
        self.last_stroke_types = []
        
        # Iterate through each point in the data
        for point_key, point_data in self.data.items():
            if not point_key.startswith("Point"):
                continue
                
            # Find all strokes in the point
            strokes = []
            self._find_strokes(point_data, strokes)
            
            if strokes:
                # Get the last stroke's type
                last_stroke = strokes[-1]
                if "StrokeType" in last_stroke:
                    self.last_stroke_types.append({
                        ##"Point": point_key,
                        "LastStrokeType": last_stroke["StrokeType"],
                        ##"Player": last_stroke.get("Player", "Unknown")
                    })
        
        return self.last_stroke_types
    
    def _find_strokes(self, data, strokes_list):
        """Helper method to recursively find all stroke dictionaries"""
        if isinstance(data, dict):
            if "StrokeType" in data:  # This is a stroke dictionary
                strokes_list.append(data)
            else:
                for value in data.values():
                    self._find_strokes(value, strokes_list)
        elif isinstance(data, list):
            for item in data:
                self._find_strokes(item, strokes_list)
    
    def save_last_stroke_types_to_csv(self, output_file="last_stroke_types.csv"):
        """Save last stroke types to CSV"""
        if not self.last_stroke_types:
            self.extract_last_stroke_types()
        
        df = pd.DataFrame(self.last_stroke_types)
        df.to_csv(output_file, index=False)
        print(f"Last stroke types saved to {output_file}")


# Example usage:
if __name__ == "__main__":
    # Initialize parsers (same as before)
    parsers = [
        parse('json/2018-27-2R16-MS-Anthony Sinisuka GINTING (INA)-Kento MOMOTA (JPN)-BLIBLI Indonesia Open-1080p.json'),
        parse('json/2018-38-5F-MS-Anthony Sinisuka GINTING (INA)-Kento MOMOTA (JPN)-Victor China Open-1080p.json'),
        parse('json/2019-04-3QF-MS-Anthony Sinisuka GINTING (INA)-Kento MOMOTA (JPN)-Daihatsu Indonesia Masters-1080p.json'),
        parse('json/2019-15-5F-MS-Anthony Sinisuka GINTING (INA)-Kento MOMOTA (JPN)-Singapore Open 2019-1080p.json'),
        parse('json/2019-20-6SF-MS-Anthony Sinisuka GINTING (INA)-Kento MOMOTA (JPN)-Sudirman Cup-1080p.json'),
        parse('json/2018-27-2R16-MS-Anthony Sinisuka GINTING (INA)-Kento MOMOTA (JPN)-BLIBLI Indonesia Open-1080p.json'),
        parse('json/2019-38-5F-MS-Anthony Sinisuka GINTING (INA)-Kento MOMOTA (JPN)-Victor China Open-1080p.json'),
        parse('json/2019-43-3QF-MS-Anthony Sinisuka GINTING (INA)-Kento Momota (JPN)-YONEX French Open-1080p-50fps.json'),
        parse('json/2019-50-5F-MS-Anthony Sinisuka GINTING (INA)-Kento MOMOTA (JPN)-HSBC BWF World Tour Finals-1080p.json')
    ]
    
    # Process each parser
    for i, parser in enumerate(parsers, 1):
        print(f"\nProcessing file {i}...")
        if parser.load_data():
            last_strokes = parser.extract_last_stroke_types()
            print(f"Found {len(last_strokes)} last stroke types")
            if last_strokes:
                print("Sample last strokes:", last_strokes[:3])  # Print first 3 for inspection
            parser.save_last_stroke_types_to_csv(f"last_stroke_types_{i}.csv")