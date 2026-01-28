import json
import os
import csv
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ResearchSystem:
    def __init__(self, start_id=0):
        # Define storage files
        self.research_storage_file = "research_responses.json"
        self.csv_storage_file = "research_responses.csv"
        
        # Load or create research storage
        self.research_responses = self.load_research_storage()
        
        # Initialize ID counter based on latest chunk
        self.current_id = self.get_latest_id() + 1
        
        # Initialize Groq client
        self.client = None
        self.initialize_groq_client()
    
    def initialize_groq_client(self):
        """Initialize Groq client"""
        try:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                print("Warning: GROQ_API_KEY not found in environment variables")
                return
            
            self.client = Groq(api_key=api_key)
            print("✓ Groq client initialized successfully")
        except Exception as e:
            print(f"Error initializing Groq client: {e}")
            self.client = None
    
    def load_research_storage(self):
        """Load existing research responses from JSON file (simple format)"""
        try:
            if os.path.exists(self.research_storage_file):
                # Check if file is empty
                if os.path.getsize(self.research_storage_file) == 0:
                    print("JSON file is empty, creating new structure")
                    return {"research_entries": []}
                
                with open(self.research_storage_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"✓ Loaded {len(data.get('research_entries', []))} existing research entries from JSON")
                    return data
            else:
                # Create empty structure
                print("Creating new JSON storage file")
                return {"research_entries": []}
        except json.JSONDecodeError:
            print("⚠️  JSON file is corrupted. Creating new one.")
            return {"research_entries": []}
        except Exception as e:
            print(f"Error loading JSON storage: {e}")
            return {"research_entries": []}
    
    def save_to_csv(self):
        """Save detailed research responses to CSV file"""
        try:
            # First check if CSV exists and has entries
            csv_entries = []
            
            if os.path.exists(self.csv_storage_file):
                # Load existing CSV entries
                with open(self.csv_storage_file, 'r', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    csv_entries = list(reader)
            
            # Get the latest entry from JSON (most recent research)
            json_entries = self.research_responses.get("research_entries", [])
            if json_entries:
                latest_entry = json_entries[-1]
                
                # Parse the text to extract query and response
                text = latest_entry.get("text", "")
                if ": " in text:
                    # Extract query and response from text
                    parts = text.split(": ", 1)
                    if len(parts) == 2:
                        query, response = parts
                        
                        # Calculate metrics
                        word_count = len(response.split())
                        char_count = len(response)
                        
                        # Create detailed CSV entry
                        csv_entry = {
                            "id": latest_entry.get("id", ""),
                            "query": query,
                            "response": response,
                            "timestamp": datetime.now().isoformat(),
                            "source": "groq_api",
                            "model": "llama-3.3-70b-versatile",
                            "chunk_size": word_count,
                            "char_count": char_count
                        }
                        
                        # Check if this ID already exists in CSV
                        existing_ids = [entry.get("id") for entry in csv_entries]
                        if str(csv_entry["id"]) not in existing_ids:
                            csv_entries.append(csv_entry)
            
            # Write all entries back to CSV
            fieldnames = ['id', 'query', 'response', 'timestamp', 'source', 'model', 'chunk_size', 'char_count']
            
            with open(self.csv_storage_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_entries)
            
            print(f"✓ CSV updated: {self.csv_storage_file}")
            print(f"  Total CSV rows: {len(csv_entries)}")
            return True
            
        except Exception as e:
            print(f"Error saving to CSV: {e}")
            return False
    
    def load_from_csv(self):
        """Load research data from CSV file"""
        try:
            if os.path.exists(self.csv_storage_file):
                entries = []
                with open(self.csv_storage_file, 'r', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        # Convert numeric fields
                        if 'chunk_size' in row and row['chunk_size']:
                            try:
                                row['chunk_size'] = int(row['chunk_size'])
                            except:
                                row['chunk_size'] = len(row.get('response', '').split())
                        if 'char_count' in row and row['char_count']:
                            try:
                                row['char_count'] = int(row['char_count'])
                            except:
                                row['char_count'] = len(row.get('response', ''))
                        entries.append(row)
                
                if entries:
                    print(f"✓ Loaded {len(entries)} entries from CSV")
                    return entries
        except Exception as e:
            print(f"Error loading from CSV: {e}")
        
        return []
    
    def convert_to_int(self, value):
        """Convert value to integer safely"""
        try:
            if isinstance(value, (int, float)):
                return int(value)
            elif isinstance(value, str):
                # Try to extract numbers from string
                import re
                numbers = re.findall(r'\d+', value)
                if numbers:
                    return int(numbers[0])
                return 0
            else:
                return 0
        except:
            return 0
    
    def get_latest_id(self):
        """Get the latest ID from existing research entries"""
        entries = self.research_responses.get("research_entries", [])
        if not entries:
            # Try to get ID from CSV as backup
            csv_data = self.load_from_csv()
            if csv_data:
                ids = []
                for entry in csv_data:
                    try:
                        id_value = entry.get("id")
                        if id_value is not None:
                            converted_id = self.convert_to_int(id_value)
                            if converted_id > 0:
                                ids.append(converted_id)
                    except:
                        continue
                if ids:
                    return max(ids)
            return 0
        
        # Extract all IDs from JSON and find the maximum
        ids = []
        for entry in entries:
            try:
                id_value = entry.get("id")
                # Convert to integer regardless of type
                if id_value is not None:
                    converted_id = self.convert_to_int(id_value)
                    if converted_id > 0:
                        ids.append(converted_id)
            except:
                continue
        
        if ids:
            return max(ids)
        return 0
    
    def get_next_id(self):
        """Get the next ID by incrementing current ID"""
        next_id = self.current_id
        self.current_id += 1  # Increment for next use
        return next_id
    
    def save_research_storage(self):
        """Save research responses to both JSON and CSV files"""
        success_json = False
        success_csv = False
        
        # Save to JSON (simple format)
        try:
            with open(self.research_storage_file, 'w', encoding='utf-8') as f:
                json.dump(self.research_responses, f, indent=2, ensure_ascii=False)
            success_json = True
        except Exception as e:
            print(f"Error saving JSON storage: {e}")
        
        # Save to CSV (detailed format)
        success_csv = self.save_to_csv()
        
        if success_json and success_csv:
            print(f"✓ Research saved to both formats:")
            print(f"  JSON (simple): {self.research_storage_file}")
            print(f"  CSV (detailed): {self.csv_storage_file}")
            print(f"  Total entries: {len(self.research_responses.get('research_entries', []))}")
            return True
        elif success_json:
            print(f"✓ Research saved to JSON only")
            return True
        elif success_csv:
            print(f"✓ Research saved to CSV only")
            return True
        else:
            print(f"❌ Failed to save research to both formats")
            return False
    
    def sync_csv_from_json(self):
        """Sync CSV file from JSON entries (useful if JSON has many entries)"""
        try:
            json_entries = self.research_responses.get("research_entries", [])
            csv_entries = []
            
            for json_entry in json_entries:
                text = json_entry.get("text", "")
                if ": " in text:
                    parts = text.split(": ", 1)
                    if len(parts) == 2:
                        query, response = parts
                        
                        # Try to get timestamp from CSV or use current
                        timestamp = datetime.now().isoformat()
                        
                        # Calculate metrics
                        word_count = len(response.split())
                        char_count = len(response)
                        
                        csv_entry = {
                            "id": json_entry.get("id", ""),
                            "query": query,
                            "response": response,
                            "timestamp": timestamp,
                            "source": "groq_api",
                            "model": "llama-3.3-70b-versatile",
                            "chunk_size": word_count,
                            "char_count": char_count
                        }
                        csv_entries.append(csv_entry)
            
            # Write to CSV
            fieldnames = ['id', 'query', 'response', 'timestamp', 'source', 'model', 'chunk_size', 'char_count']
            
            with open(self.csv_storage_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_entries)
            
            print(f"✓ Synced CSV from JSON: {len(csv_entries)} entries")
            return True
            
        except Exception as e:
            print(f"Error syncing CSV: {e}")
            return False
    
    def send_research_request(self, user_query):
        """Send research request to Groq API and get response"""
        if not self.client:
            return None, "Groq client not initialized"
        
        try:
            # Create research prompt
            research_prompt = f'''Analyze the user's query and respond appropriately

User Query: "{user_query}"

**Response Guidelines:**
1. **Assess Complexity:** Determine if this is a simple greeting, basic question, or requires in-depth explanation
2. **Match Depth:** Provide depth proportional to query complexity
3. **Natural Tone:** Write in clear, human-like language

**Response Framework:**
- For greetings/simple queries: Brief, warm, conversational response
- For basic factual questions: Concise answer with 2-3 key points
- For complex topics: Detailed explanation with examples, context, and structure
- Always maintain helpful, informative tone

**Formatting:** Use natural paragraphs, no markdown unless needed for clarity. Adjust length from 1-2 sentences to multiple paragraphs based on query depth.'''

            print(f"📡 Sending request to Groq API...")
            
            # Send to Groq
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": research_prompt,
                    }
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=800
            )
            
            response = chat_completion.choices[0].message.content
            print(f"✓ Received response from Groq API")
            return response, None
            
        except Exception as e:
            error_msg = f"Error calling Groq API: {e}"
            print(error_msg)
            return None, error_msg
    
    def receive_and_save_research(self, user_query):
        """Main function: send research, receive response, and save to JSON & CSV"""
        if not self.client:
            return None, "Groq client not initialized"
        
        # Get next ID
        research_id = self.get_next_id()
        
        print(f"\n{'='*50}")
        print(f"🧠 Starting research with ID: {research_id}")
        print(f"📝 Query: {user_query}")
        print(f"{'='*50}")
        
        # Send research request
        research_response, error = self.send_research_request(user_query)
        
        if error:
            return None, error
        
        # Create SIMPLE JSON entry (query + response combined)
        combined_text = f'{user_query}: {research_response}'
        json_entry = {
            "id": research_id,
            "text": combined_text
        }
        
        # Add to JSON storage
        self.research_responses["research_entries"].append(json_entry)
        
        # Save to both JSON and CSV
        save_success = self.save_research_storage()
        
        if not save_success:
            return research_response, "Research completed but failed to save to storage"
        
        print(f"✅ Research saved successfully!")
        print(f"   ID: {research_id}")
        print(f"   Response length: {len(research_response)} characters")
        print(f"   Word count: {len(research_response.split())}")
        print(f"   JSON saved: {self.research_storage_file}")
        print(f"   CSV saved: {self.csv_storage_file}")
        
        return research_response, None
    
    def get_recent_research(self, limit=5, format="json"):
        """Get recent research entries in specified format"""
        if format == "json":
            entries = self.research_responses.get("research_entries", [])
            if not entries:
                return []
            
            # Sort entries by ID
            sorted_entries = sorted(entries,
                                   key=lambda x: self.convert_to_int(x.get("id", 0)),
                                   reverse=True)
            return sorted_entries[:limit]
        
        elif format == "csv":
            # Get from CSV
            csv_data = self.load_from_csv()
            if not csv_data:
                return []
            
            # Sort by ID
            sorted_entries = sorted(csv_data,
                                   key=lambda x: self.convert_to_int(x.get("id", 0)),
                                   reverse=True)
            return sorted_entries[:limit]
    
    def search_research(self, keyword, format="csv"):
        """Search research by keyword"""
        results = []
        
        if format == "json":
            for entry in self.research_responses.get("research_entries", []):
                text = entry.get("text", "")
                if keyword.lower() in text.lower():
                    results.append(entry)
        
        elif format == "csv":
            csv_data = self.load_from_csv()
            for entry in csv_data:
                if (keyword.lower() in entry.get("query", "").lower() or 
                    keyword.lower() in entry.get("response", "").lower()):
                    results.append(entry)
        
        # Sort results by ID (newest first)
        return sorted(results,
                     key=lambda x: self.convert_to_int(x.get("id", 0)),
                     reverse=True)
    
    def print_research_summary(self):
        """Print summary of all research"""
        json_entries = self.research_responses.get("research_entries", [])
        csv_data = self.load_from_csv()
        
        print(f"\n{'='*50}")
        print(f"📊 Research Database Summary")
        print(f"{'='*50}")
        print(f"JSON entries (simple): {len(json_entries)}")
        print(f"CSV entries (detailed): {len(csv_data)}")
        print(f"Current ID counter: {self.current_id}")
        print(f"Latest ID: {self.get_latest_id()}")
        
        if json_entries:
            print(f"\nRecent JSON entries (simple format):")
            recent = self.get_recent_research(3, format="json")
            for entry in recent:
                entry_id = entry.get("id", "N/A")
                text = entry.get("text", "")
                if len(text) > 60:
                    text = text[:57] + "..."
                print(f"  📌 ID {entry_id} | {text}")
        
        if csv_data:
            print(f"\nRecent CSV entries (detailed format):")
            recent_csv = self.get_recent_research(3, format="csv")
            for entry in recent_csv:
                entry_id = entry.get("id", "N/A")
                query = entry.get("query", "")
                if len(query) > 40:
                    query = query[:37] + "..."
                timestamp = entry.get("timestamp", "")
                if timestamp and len(timestamp) > 10:
                    timestamp = timestamp[:16]  # Show date and time
                print(f"  📌 ID {entry_id} | Q: {query}")
                print(f"    ⏰ {timestamp} | 📝 {entry.get('chunk_size', 0)} words")
    
    def print_json_preview(self, num_entries=3):
        """Print preview of JSON file"""
        entries = self.research_responses.get("research_entries", [])
        if not entries:
            print("No JSON entries yet.")
            return
        
        print(f"\n📄 JSON File Preview ({self.research_storage_file}):")
        print("-" * 80)
        
        for i, entry in enumerate(entries[-num_entries:]):  # Show latest entries
            entry_id = entry.get("id", "N/A")
            text = entry.get("text", "")
            
            # Split text into query and response
            if ": " in text:
                query, response = text.split(": ", 1)
                print(f"ID {entry_id}:")
                print(f"  Query: {query[:50]}..." if len(query) > 50 else f"  Query: {query}")
                print(f"  Response: {response[:80]}..." if len(response) > 80 else f"  Response: {response}")
            else:
                print(f"ID {entry_id}: {text[:100]}...")
            
            if i < len(entries[-num_entries:]) - 1:
                print("-" * 40)
        
        print(f"\nTotal JSON entries: {len(entries)}")
    
    def print_csv_preview(self, num_rows=3):
        """Print preview of CSV file"""
        try:
            if not os.path.exists(self.csv_storage_file):
                print(f"CSV file {self.csv_storage_file} does not exist yet.")
                return
            
            print(f"\n📋 CSV File Preview ({self.csv_storage_file}):")
            print("-" * 80)
            
            with open(self.csv_storage_file, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                
                # Print header
                headers = next(reader)
                print(f"Headers: {', '.join(headers)}")
                print("-" * 80)
                
                # Print last few rows (most recent)
                all_rows = list(reader)
                start_idx = max(0, len(all_rows) - num_rows)
                
                for i in range(start_idx, len(all_rows)):
                    row = all_rows[i]
                    display_row = []
                    for j, cell in enumerate(row):
                        if headers[j] == 'response' and len(cell) > 40:
                            display_row.append(cell[:37] + "...")
                        elif headers[j] == 'query' and len(cell) > 30:
                            display_row.append(cell[:27] + "...")
                        elif len(cell) > 20:
                            display_row.append(cell[:17] + "...")
                        else:
                            display_row.append(cell)
                    
                    print(f"Row {i+1}: {display_row}")
            
            print(f"-" * 80)
            print(f"Total CSV rows: {len(all_rows)}")
            
        except Exception as e:
            print(f"Error reading CSV file: {e}")

# Create global instance
self_research = ResearchSystem()

# Test the system
if __name__ == "__main__":
    print("🔧 Testing ResearchSystem with dual format storage...")
    
    # Print initial state
    self_research.print_research_summary()
    
    # Sync CSV if JSON has entries but CSV is empty
    if (len(self_research.research_responses.get("research_entries", [])) > 0 and 
        not os.path.exists(self_research.csv_storage_file)):
        print("\n🔄 Syncing existing JSON entries to CSV...")
        self_research.sync_csv_from_json()
    
    # Test a research query
    if self_research.client:
        test_queries = [
            "What is artificial intelligence?",
            "Explain machine learning",
            "What are neural networks?"
        ]
        
        for i, test_query in enumerate(test_queries):
            print(f"\n🚀 Test {i+1}/{len(test_queries)}: {test_query}")
            
            answer, error = self_research.receive_and_save_research(test_query)
            
            if error:
                print(f"❌ Error: {error}")
            else:
                print(f"✅ Research completed!")
        
        # Show previews
        print(f"\n{'='*60}")
        print("📊 FINAL STORAGE PREVIEW")
        print(f"{'='*60}")
        
        self_research.print_json_preview(2)
        self_research.print_csv_preview(2)
        
        # Show final summary
        self_research.print_research_summary()
        
    else:
        print("❌ Groq client not initialized. Check your GROQ_API_KEY in .env file")
        print("Create a .env file with: GROQ_API_KEY=your_api_key_here")
