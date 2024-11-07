import requests
import json
from datetime import datetime
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ClaimProcessor:
    def __init__(self, api_url: str, api_key: str):
        """
        Initialize the claim processor with API credentials.

        Args:
            api_url (str): Base URL for the insurance API
            api_key (str): Authentication key for API access
        """
        self.api_url = api_url
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

    def fetch_claim_data(self, claim_id: str) -> Optional[Dict]:
        """
        Fetch claim data from the API.

        Args:
            claim_id (str): Unique identifier for the claim

        Returns:
            Optional[Dict]: Claim data if successful, None if failed
        """
        try:
            response = requests.get(
                f'{self.api_url}/claims/{claim_id}',
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching claim data: {str(e)}")
            return None

    def process_claim(self, claim_data: Dict) -> Dict:
        """
        Process the raw claim data and extract relevant information.

        Args:
            claim_data (Dict): Raw claim data from API

        Returns:
            Dict: Processed claim information
        """
        processed_data = {
            'claim_number': claim_data.get('claim_number'),
            'status': claim_data.get('status'),
            'filing_date': datetime.fromisoformat(claim_data.get('filing_date')),
            'total_amount': float(claim_data.get('total_amount', 0)),
            'claimant_info': {
                'name': claim_data.get('claimant', {}).get('name'),
                'policy_number': claim_data.get('claimant', {}).get('policy_number'),
                'contact': claim_data.get('claimant', {}).get('contact')
            }
        }

        # Calculate claim age in days
        processed_data['claim_age'] = (datetime.now() - processed_data['filing_date']).days

        # Add priority flag for claims over 30 days
        processed_data['is_priority'] = processed_data['claim_age'] > 30

        return processed_data

    def validate_claim(self, processed_data: Dict) -> List[str]:
        """
        Validate the processed claim data.

        Args:
            processed_data (Dict): Processed claim data

        Returns:
            List[str]: List of validation errors, empty if valid
        """
        errors = []

        required_fields = ['claim_number', 'status', 'filing_date', 'total_amount']
        for field in required_fields:
            if not processed_data.get(field):
                errors.append(f"Missing required field: {field}")

        if processed_data.get('total_amount', 0) <= 0:
            errors.append("Invalid claim amount")

        if not all(processed_data['claimant_info'].values()):
            errors.append("Incomplete claimant information")

        return errors

def main():
    # Example usage
    api_url = "https://api.insurance-company.com/v1"
    api_key = "your_api_key_here"
    claim_id = "CLAIM123"

    processor = ClaimProcessor(api_url, api_key)

    # Fetch and process claim
    raw_data = processor.fetch_claim_data(claim_id)
    if not raw_data:
        logger.error("Failed to fetch claim data")
        return

    # Process the claim data
    processed_data = processor.process_claim(raw_data)

    # Validate the processed data
    validation_errors = processor.validate_claim(processed_data)
    if validation_errors:
        logger.error("Validation errors found:")
        for error in validation_errors:
            logger.error(f"- {error}")
        return

    # Log successful processing
    logger.info(f"Successfully processed claim {claim_id}")
    logger.info(f"Claim age: {processed_data['claim_age']} days")
    logger.info(f"Priority status: {'Yes' if processed_data['is_priority'] else 'No'}")

if __name__ == "__main__":
    main()