# services/gemini_recommendation_service.py
import google.generativeai as genai
import json

class GeminiRecommendationService:
    def __init__(self, api_key):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
    
    def generate_insights(self, bill_data, predictions, anomalies):
        """Generate AI insights based on bill data, predictions, and anomalies"""
        try:
            # Prepare the prompt with bill data and predictions
            prompt = f"""
            As an electricity bill analysis expert, provide personalized energy-saving recommendations based on this data:

            Current Bill:
            - Account: {bill_data.get('account_number')}
            - kWh Used: {bill_data.get('kwh_used')}
            - Bill Amount: ${bill_data.get('total_bill_amount')}
            - Average Daily Usage: {bill_data.get('avg_daily_usage')} kWh
            - Billing Period: {bill_data.get('billing_start_date')} to {bill_data.get('billing_end_date')}
            
            Future Predictions:
            {json.dumps(predictions, indent=2)}
            
            Anomalies Detected:
            {json.dumps(anomalies, indent=2) if anomalies else "No anomalies detected."}

            Provide 3-5 specific, actionable recommendations to help reduce electricity costs.
            Focus on practical tips based on usage patterns, seasonal changes, and potential savings.
            Format your response as a JSON array of recommendation objects with 'title' and 'description' fields.
            """

            # Get AI-generated recommendations
            response = self.model.generate_content(prompt)
            
            # Parse the response to extract JSON recommendations
            text = response.text
            
            # Find the JSON array in the response
            import re
            json_pattern = r'(\[[\s\S]*\])'
            match = re.search(json_pattern, text)
            
            if match:
                json_str = match.group(1)
                try:
                    recommendations = json.loads(json_str)
                    return recommendations
                except json.JSONDecodeError:
                    # Fallback to a simple structure if JSON parsing fails
                    return [{"title": "Energy Saving Tip", "description": text}]
            else:
                # Fallback if no JSON found
                return [{"title": "Energy Saving Tip", "description": text}]
                
        except Exception as e:
            print(f"Error generating AI insights: {str(e)}")
            return [{"title": "Energy Saving Tip", "description": "Consider energy-efficient appliances and turning off lights when not in use."}]