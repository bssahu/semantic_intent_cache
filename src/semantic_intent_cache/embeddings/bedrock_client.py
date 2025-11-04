"""Bedrock client for AWS services."""

import asyncio
import json
import logging

import boto3
from botocore.config import Config as BotoConfig

logger = logging.getLogger(__name__)


class BedrockClient:
    """Client for interacting with AWS Bedrock."""

    def __init__(
        self,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_region: str = "us-east-1",
    ):
        """
        Initialize the Bedrock client.

        Args:
            aws_access_key_id: AWS access key ID (optional, uses credentials chain if not provided).
            aws_secret_access_key: AWS secret access key (optional, uses credentials chain if not provided).
            aws_region: AWS region for Bedrock.
        """
        try:
            # Debug logging for credentials
            logger.info("=== BEDROCK CLIENT INIT DEBUG ===")
            logger.info(f"Config.AWS_REGION: {aws_region}")
            if aws_access_key_id:
                logger.info(f"Config.AWS_ACCESS_KEY_ID: {aws_access_key_id}")
            if aws_secret_access_key:
                logger.info(
                    f"Config.AWS_SECRET_ACCESS_KEY: {aws_secret_access_key[:10] if aws_secret_access_key else 'None'}..."
                )

            # Configure boto3 with SSL fix for production environments
            boto_config = BotoConfig(
                region_name=aws_region,
                retries={"max_attempts": 3, "mode": "adaptive"},
                # SSL configuration for production environments
                tcp_keepalive=True,
                max_pool_connections=50,
            )

            # Initialize client with explicit credentials or use default credential chain
            client_kwargs = {
                "service_name": "bedrock-runtime",
                "region_name": aws_region,
                "config": boto_config,
            }

            if aws_access_key_id and aws_secret_access_key:
                client_kwargs["aws_access_key_id"] = aws_access_key_id
                client_kwargs["aws_secret_access_key"] = aws_secret_access_key

            self.client = boto3.client(**client_kwargs)

            logger.info("Successfully initialized Bedrock client")
            logger.info("=====================================")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {str(e)}")
            raise

    def invoke_model(
        self,
        model_id: str,
        body: str | bytes,
        max_tokens: int = 2048,
        agent_name: str = "Unknown",
    ) -> dict:
        """
        Invoke a Bedrock model.

        Args:
            model_id: The Bedrock model ID to invoke.
            body: Request body (JSON string or bytes).
            max_tokens: Maximum tokens for the response.
            agent_name: Name for logging purposes.

        Returns:
            Response dictionary from Bedrock.
        """
        try:
            # Log the model being invoked
            logger.info(f"🚀 [{agent_name}] Invoking Bedrock model: {model_id}")

            # Convert body to bytes if string
            if isinstance(body, str):
                body_bytes = body.encode("utf-8")
            else:
                body_bytes = body

            logger.info(f"📝 [{agent_name}] Request body size: {len(body_bytes)} characters")
            logger.info(f"⚙️  [{agent_name}] Max tokens: {max_tokens}")

            # Invoke model
            response = self.client.invoke_model(
                modelId=model_id,
                body=body_bytes,
                contentType="application/json",
            )

            return response
        except Exception as e:
            error_msg = str(e)
            error_str_lower = error_msg.lower()
            
            # Check for SSL certificate errors
            if "certificate" in error_str_lower or "ssl" in error_str_lower:
                logger.error(
                    f"Error invoking Bedrock model {model_id}: {error_msg}\n"
                    f"SSL Certificate Error detected. This is often caused by missing CA certificates.\n"
                    f"On macOS: Run 'brew install ca-certificates' or update your certificates.\n"
                    f"On Linux: Ensure ca-certificates package is installed.\n"
                    f"Alternatively, set REQUESTS_CA_BUNDLE or CURL_CA_BUNDLE environment variable."
                )
            # Check for inference profile requirement or access issues
            elif "inference profile" in error_str_lower or "validationexception" in error_str_lower:
                logger.error(
                    f"Error invoking Bedrock model {model_id}: {error_msg}\n"
                    f"\n"
                    f"This error typically means:\n"
                    f"  1. Model access not enabled in your AWS account\n"
                    f"  2. Model requires an inference profile (newer models)\n"
                    f"  3. On-demand access not available for this model\n"
                    f"\n"
                    f"To fix this:\n"
                    f"  • Go to AWS Bedrock Console → Model access → Enable Anthropic models\n"
                    f"  • Request access to on-demand models (if available in your region)\n"
                    f"  • Try a model that typically supports on-demand:\n"
                    f"    - anthropic.claude-3-haiku-20240307-v1:0 (faster, cheaper)\n"
                    f"    - anthropic.claude-3-sonnet-20240229-v1:0 (older Sonnet)\n"
                    f"  • Or use an inference profile ARN format:\n"
                    f"    arn:aws:bedrock:REGION::inference-profile/MODEL_ID\n"
                    f"  • Check available models: AWS Console → Bedrock → Foundation models"
                )
            else:
                logger.error(f"Error invoking Bedrock model {model_id}: {error_msg}")
            raise

    async def invoke_model_async(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 2048,
        agent_name: str = "Unknown",
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Invoke the model asynchronously using Messages API.

        Args:
            model_id: The Bedrock model ID or inference profile ARN to invoke.
            prompt: The prompt text to send to the model.
            max_tokens: Maximum tokens for the response.
            agent_name: Name for logging purposes.
            temperature: Sampling temperature (0.0 to 1.0).
            top_p: Nucleus sampling parameter (0.0 to 1.0).

        Returns:
            The text completion from the model.
        """
        try:
            # Log the model being invoked
            logger.info(f"🚀 [{agent_name}] Invoking Bedrock model: {model_id}")
            logger.info(f"📝 [{agent_name}] Prompt length: {len(prompt)} characters")
            logger.info(f"⚙️  [{agent_name}] Max tokens: {max_tokens}")

            # Create the request body for Messages API
            body = json.dumps({
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "anthropic_version": "bedrock-2023-05-31"
            })

            logger.debug(f"📤 [{agent_name}] Request body size: {len(body)} characters")

            # Run the API call in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.invoke_model(
                    modelId=model_id,
                    body=body.encode("utf-8"),
                    contentType="application/json"
                )
            )

            # Parse the response
            response_body = json.loads(response['body'].read())
            completion = response_body.get('content', [{}])[0].get('text', '')

            logger.info(f"✅ [{agent_name}] Bedrock model response received")
            logger.info(f"📊 [{agent_name}] Response length: {len(completion)} characters")
            logger.debug(
                f"📄 [{agent_name}] Response preview: {completion[:200]}..."
                if len(completion) > 200
                else f"📄 [{agent_name}] Full response: {completion}"
            )

            return completion.strip()

        except Exception as e:
            logger.error(f"❌ [{agent_name}] Error invoking Bedrock model: {str(e)}")
            logger.error(f"🔍 [{agent_name}] Model ID that failed: {model_id}")
            logger.error(f"🔍 [{agent_name}] Error type: {type(e).__name__}")
            
            error_msg = str(e)
            error_str_lower = error_msg.lower()
            
            # Check for SSL certificate errors
            if "certificate" in error_str_lower or "ssl" in error_str_lower:
                logger.error(
                    f"SSL Certificate Error detected. This is often caused by missing CA certificates.\n"
                    f"On macOS: Run 'brew install ca-certificates' or update your certificates.\n"
                    f"On Linux: Ensure ca-certificates package is installed.\n"
                    f"Alternatively, set REQUESTS_CA_BUNDLE or CURL_CA_BUNDLE environment variable."
                )
            # Check for inference profile requirement or access issues
            elif "inference profile" in error_str_lower or "validationexception" in error_str_lower:
                logger.error(
                    f"This error typically means:\n"
                    f"  1. Model access not enabled in your AWS account\n"
                    f"  2. Model requires an inference profile (newer models)\n"
                    f"  3. On-demand access not available for this model\n"
                    f"\n"
                    f"To fix this:\n"
                    f"  • Go to AWS Bedrock Console → Model access → Enable Anthropic models\n"
                    f"  • Request access to on-demand models (if available in your region)\n"
                    f"  • Try a model that typically supports on-demand:\n"
                    f"    - anthropic.claude-3-haiku-20240307-v1:0 (faster, cheaper)\n"
                    f"    - anthropic.claude-3-sonnet-20240229-v1:0 (older Sonnet)\n"
                    f"  • Or use an inference profile ARN format:\n"
                    f"    arn:aws:bedrock:REGION::inference-profile/MODEL_ID\n"
                    f"  • Check available models: AWS Console → Bedrock → Foundation models"
                )
            
            raise
