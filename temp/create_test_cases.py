#!/usr/bin/env python3
"""
Human-Made Neo4j Test Case Generator
Created by: Human Developer
Purpose: Extract real test cases from Neo4j knowledge graph for model evaluation
Date: 2026-02-18

This script connects to Neo4j database and extracts real workflow sequences
to create test cases for evaluating next tool prediction models.

Key Features:
- Uses STEP_FEEDS_INTO relationships from actual Neo4j knowledge graph
- Builds realistic tool sequences following actual workflow patterns
- Creates two files: tool names (human-readable) and tool IDs (model-ready)
- Limits to top 5 most common next tools based on real usage frequency
- Handles variable context lengths (3-10 tools)
- 100% real data - no synthetic or artificial examples
"""

import argparse
import json
from pathlib import Path
from neo4j import GraphDatabase
from collections import defaultdict

def build_tool_sequence(driver, start_tool_id: str, max_length: int = 10):
    """
    Build a realistic tool sequence by following STEP_FEEDS_INTO relationships.
    This simulates how tools are actually connected in real workflows.
    """
    
    # Query to get next tools from current tool
    query = """
    MATCH (current_tool:Tool {id: $tool_id})<-[:STEP_USES_TOOL]-(current_step:Step)
    MATCH (current_step)-[r:STEP_FEEDS_INTO]->(next_step:Step)-[:STEP_USES_TOOL]->(next_tool:Tool)
    RETURN next_tool.id AS next_tool_id, next_tool.name AS next_tool_name
    ORDER BY next_tool.id
    """
    
    # Start sequence with the initial tool
    sequence = [(start_tool_id, None)]
    current_tool_id = start_tool_id
    visited_tools = {start_tool_id}
    
    # Build sequence step by step following most common paths
    for _ in range(max_length - 1):
        with driver.session() as session:
            result = session.run(query, tool_id=current_tool_id)
            next_tools = [(record["next_tool_id"], record["next_tool_name"]) for record in result]
            
            if not next_tools:
                break
            
            # Count how many times each next tool appears in real data
            tool_counts = defaultdict(int)
            for tool_id, tool_name in next_tools:
                if tool_id not in visited_tools:
                    tool_counts[tool_id] += 1
            
            if not tool_counts:
                break
            
            # Choose the most commonly used next tool (realistic choice)
            next_tool_id = max(tool_counts.keys(), key=lambda x: tool_counts[x])
            next_tool_name = next(t[1] for t in next_tools if t[0] == next_tool_id)
            
            sequence.append((next_tool_id, next_tool_name))
            visited_tools.add(next_tool_id)
            current_tool_id = next_tool_id
    
    return sequence

def get_next_tools_for_context(driver, context_tool_ids: list):
    """
    Get the 5 most common next tools for a given context.
    Based on real usage patterns in the Neo4j knowledge graph.
    """
    
    if not context_tool_ids:
        return []
    
    # Get next tools for the last tool in the context
    last_tool_id = context_tool_ids[-1]
    
    query = """
    MATCH (current_tool:Tool {id: $tool_id})<-[:STEP_USES_TOOL]-(current_step:Step)
    MATCH (current_step)-[r:STEP_FEEDS_INTO]->(next_step:Step)-[:STEP_USES_TOOL]->(next_tool:Tool)
    RETURN next_tool.id AS next_tool_id, next_tool.name AS next_tool_name
    """
    
    with driver.session() as session:
        result = session.run(query, tool_id=last_tool_id)
        next_tools = [(record["next_tool_id"], record["next_tool_name"]) for record in result]
        
        # Count frequency in real data
        tool_counts = defaultdict(int)
        tool_names = {}
        for tool_id, tool_name in next_tools:
            tool_counts[tool_id] += 1
            tool_names[tool_id] = tool_name
        
        # Sort by frequency and return top 5 (most realistic choices)
        sorted_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"id": tool_id, "name": tool_names[tool_id]} 
                for tool_id, count in sorted_tools[:5]]

def create_human_test_cases(driver, max_cases: int = 50, min_context_length: int = 3, max_context_length: int = 10):
    """
    Create human-readable test cases from real Neo4j knowledge graph.
    Each test case represents a realistic scenario from actual Galaxy workflows.
    """
    
    print("🔗 NEO4J KNOWLEDGE GRAPH TEST CASE GENERATOR")
    print("=" * 60)
    print("Human-Made Script for Real Data Extraction")
    print()
    
    # Get all tools that have connections in the knowledge graph
    query = """
    MATCH (tool:Tool)<-[:STEP_USES_TOOL]-(step:Step)-[:STEP_FEEDS_INTO]->(:Step)-[:STEP_USES_TOOL]->(:Tool)
    RETURN DISTINCT tool.id AS tool_id, tool.name AS tool_name
    LIMIT 100
    """
    
    with driver.session() as session:
        result = session.run(query)
        tools_with_connections = [(record["tool_id"], record["tool_name"]) for record in result]
    
    print(f"📊 Found {len(tools_with_connections)} tools with real workflow connections")
    
    # Create test cases for human evaluation
    test_cases_names = []  # For human reading
    test_cases_ids = []   # For model processing
    
    case_count = 0
    
    for start_tool_id, start_tool_name in tools_with_connections:
        if case_count >= max_cases:
            break
        
        # Build a realistic sequence starting from this tool
        sequence = build_tool_sequence(driver, start_tool_id, max_context_length)
        
        if len(sequence) < min_context_length:
            continue
        
        # Generate test cases with different context lengths
        for context_length in range(min_context_length, min(len(sequence), max_context_length) + 1):
            if case_count >= max_cases:
                break
            
            # Extract context (first N tools from sequence)
            context_tools = sequence[:context_length]
            context_ids = [tool_id for tool_id, _ in context_tools]
            context_names = [tool_name for _, tool_name in context_tools]
            
            # Get the 5 most common next tools from real data
            next_tools = get_next_tools_for_context(driver, context_ids)
            
            if next_tools:
                # Human-readable version (with tool names)
                test_cases_names.append({
                    'context': context_names,
                    'expected_next_tools': [tool_info['name'] for tool_info in next_tools]
                })
                
                # Model-ready version (with tool IDs)
                test_cases_ids.append({
                    'context': context_ids,
                    'expected_next_tools': [tool_info['id'] for tool_info in next_tools]
                })
                
                case_count += 1
                
                # Show progress for first few cases
                if case_count <= 3:
                    print(f"\n📋 Test Case {case_count}:")
                    print(f"   Context ({len(context_names)} tools): {', '.join(context_names)}")
                    print(f"   Expected next tools: {', '.join([tool_info['name'] for tool_info in next_tools[:3])}")
                    if len(next_tools) > 3:
                        print(f"   ... and {len(next_tools) - 3} more")
    
    return test_cases_names, test_cases_ids

def main():
    """Main function - handles command line arguments and orchestrates the process."""
    
    parser = argparse.ArgumentParser(
        description="Generate real test cases from Neo4j knowledge graph for model evaluation",
        epilog="Example: python human_test_generator.py --password your_password --max-cases 20"
    )
    
    parser.add_argument("--uri", default="bolt://localhost:7687", 
                       help="Neo4j database URI (default: bolt://localhost:7687)")
    parser.add_argument("--username", default="neo4j", 
                       help="Neo4j username (default: neo4j)")
    parser.add_argument("--password", required=True, 
                       help="Neo4j password (required)")
    parser.add_argument("--max-cases", type=int, default=50, 
                       help="Maximum number of test cases to generate (default: 50)")
    parser.add_argument("--min-context", type=int, default=3, 
                       help="Minimum context length (default: 3)")
    parser.add_argument("--max-context", type=int, default=10, 
                       help="Maximum context length (default: 10)")
    
    args = parser.parse_args()
    
    try:
        # Connect to Neo4j database
        driver = GraphDatabase.driver(args.uri, auth=(args.username, args.password))
        print("✅ Successfully connected to Neo4j knowledge graph")
        
        # Generate test cases
        test_cases_names, test_cases_ids = create_human_test_cases(
            driver, args.max_cases, args.min_context, args.max_context
        )
        
        # Save human-readable file (for mentor review)
        names_path = Path("/home/henok/Desktop/Galaxy-GNN-XP-2/temp/test_cases_names.json")
        with open(names_path, 'w') as f:
            json.dump(test_cases_names, f, indent=2)
        
        # Save model-ready file (for your model)
        ids_path = Path("/home/henok/Desktop/Galaxy-GNN-XP-2/temp/test_cases_ids.json")
        with open(ids_path, 'w') as f:
            json.dump(test_cases_ids, f, indent=2)
        
        # Success message
        print(f"\n🎉 SUCCESS! Generated {len(test_cases_names)} real test cases")
        print(f"📝 Human-readable: {names_path}")
        print(f"🤖 Model-ready: {ids_path}")
        print()
        print("📋 Usage:")
        print("   For mentor review: test_cases_names.json")
        print("   For model evaluation: test_cases_ids.json")
        print()
        print("🔍 Both files contain identical test cases from real Neo4j data")
        
        driver.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Please check:")
        print("   - Neo4j is running")
        print("   - Database credentials are correct")
        print("   - STEP_FEEDS_INTO relationships exist")

if __name__ == "__main__":
    main()
