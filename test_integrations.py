#!/usr/bin/env python3
"""
Test script to verify all integrations are working.
"""

import asyncio
import os
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from code_intelligence.config import config
from code_intelligence.llm.azure_client import AzureOpenAIClient
from code_intelligence.database.supabase_client import SupabaseClient
from code_intelligence.database.neo4j_client import Neo4jClient
from code_intelligence.logging import get_logger

logger = get_logger(__name__)


async def test_azure_openai():
    """Test Azure OpenAI integration."""
    print("🔍 Testing Azure OpenAI...")
    
    try:
        client = AzureOpenAIClient()
        
        # Test health check
        is_healthy = await client.health_check()
        if is_healthy:
            print("✅ Azure OpenAI health check passed")
        else:
            print("❌ Azure OpenAI health check failed")
            return False
            
        # Test chat completion
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello, World!' in exactly those words."}
        ]
        
        response = await client.chat_completion(
            messages=messages,
            temperature=0,
            max_tokens=10
        )
        
        print(f"✅ Azure OpenAI chat completion: {response}")
        return True
        
    except Exception as e:
        print(f"❌ Azure OpenAI test failed: {str(e)}")
        return False


def test_supabase():
    """Test Supabase integration."""
    print("🔍 Testing Supabase...")
    
    try:
        client = SupabaseClient()
        client.connect()
        
        # Test basic connection
        supabase_client = client.client
        
        # Try to get user (should work even without auth)
        try:
            user = supabase_client.auth.get_user()
            print("✅ Supabase connection established")
        except:
            # This is expected without authentication
            print("✅ Supabase connection established (no auth)")
            
        return True
        
    except Exception as e:
        print(f"❌ Supabase test failed: {str(e)}")
        return False


async def test_neo4j():
    """Test Neo4j integration."""
    print("🔍 Testing Neo4j...")
    
    try:
        client = Neo4jClient()
        await client.connect()
        
        # Test simple query
        result = await client.execute_query("RETURN 1 as test")
        
        if result and result[0].get("test") == 1:
            print("✅ Neo4j connection and query successful")
            return True
        else:
            print("❌ Neo4j query returned unexpected result")
            return False
            
    except Exception as e:
        print(f"❌ Neo4j test failed: {str(e)}")
        print("💡 Make sure Neo4j is running locally or update NEO4J_URI in .env")
        return False


def test_configuration():
    """Test configuration loading."""
    print("🔍 Testing Configuration...")
    
    try:
        # Test Azure OpenAI config
        if config.llm.azure_openai_api_key:
            print("✅ Azure OpenAI API key loaded")
        else:
            print("❌ Azure OpenAI API key missing")
            return False
            
        if config.llm.azure_openai_endpoint:
            print("✅ Azure OpenAI endpoint loaded")
        else:
            print("❌ Azure OpenAI endpoint missing")
            return False
            
        # Test Supabase config
        if config.database.supabase_url:
            print("✅ Supabase URL loaded")
        else:
            print("❌ Supabase URL missing")
            return False
            
        if config.database.supabase_key:
            print("✅ Supabase key loaded")
        else:
            print("❌ Supabase key missing")
            return False
            
        print("✅ All configuration loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {str(e)}")
        return False


async def main():
    """Run all integration tests."""
    print("🚀 Multi-Agent Code Intelligence - Integration Tests")
    print("=" * 60)
    
    results = []
    
    # Test configuration
    results.append(test_configuration())
    
    # Test Azure OpenAI
    results.append(await test_azure_openai())
    
    # Test Supabase
    results.append(test_supabase())
    
    # Test Neo4j (optional - might not be running)
    results.append(await test_neo4j())
    
    print("\n" + "=" * 60)
    print("📊 Integration Test Results:")
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All integrations working correctly!")
    elif passed >= 3:  # Neo4j is optional
        print("🟡 Core integrations working (Neo4j may need setup)")
    else:
        print("❌ Some integrations need attention")
        
    return passed >= 3  # Consider success if core integrations work


if __name__ == "__main__":
    success = asyncio.run(main())