#!/usr/bin/env python3
"""Debug utilities for memory processing logging."""

import asyncio
import json
import logging
import os
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Setup debug logger
debug_logger = logging.getLogger("memory_debug")
debug_logger.setLevel(logging.INFO)

# Debug data storage
DEBUG_DIR = Path("./debug_data")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_FILE = DEBUG_DIR / "memory_processing.json"
MAX_DEBUG_ENTRIES = 1000  # Keep last 1000 entries

# Thread lock for safe file writing
_debug_lock = threading.Lock()

class MemoryProcessingDebugger:
    """Handles debug logging for memory processing operations."""
    
    def __init__(self):
        self.debug_entries: List[Dict[str, Any]] = []
        self._load_existing_entries()
    
    def _load_existing_entries(self):
        """Load existing debug entries from file."""
        try:
            if DEBUG_FILE.exists():
                with open(DEBUG_FILE, 'r') as f:
                    self.debug_entries = json.load(f)
                    # Keep only the most recent entries
                    if len(self.debug_entries) > MAX_DEBUG_ENTRIES:
                        self.debug_entries = self.debug_entries[-MAX_DEBUG_ENTRIES:]
                        self._save_to_file()
                debug_logger.info(f"Loaded {len(self.debug_entries)} existing debug entries")
            else:
                self.debug_entries = []
                debug_logger.info("No existing debug file found, starting fresh")
        except Exception as e:
            debug_logger.error(f"Error loading debug entries: {e}")
            self.debug_entries = []
    
    def _save_to_file(self):
        """Save debug entries to file in a thread-safe manner."""
        try:
            with _debug_lock:
                with open(DEBUG_FILE, 'w') as f:
                    json.dump(self.debug_entries, f, indent=2)
        except Exception as e:
            debug_logger.error(f"Error saving debug entries: {e}")
    
    def log_memory_processing(
        self,
        user_id: str,
        audio_uuid: str,
        transcript_text: str,
        memories_created: Optional[List[Dict[str, Any]]] = None,
        action_items_created: Optional[List[Dict[str, Any]]] = None,
        processing_success: bool = True,
        error_message: Optional[str] = None,
        processing_time_ms: Optional[float] = None
    ):
        """Log a memory processing operation."""
        try:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "unix_timestamp": int(time.time()),
                "user_id": user_id,
                "audio_uuid": audio_uuid,
                "transcript_text": transcript_text[:500] + "..." if len(transcript_text) > 500 else transcript_text,
                "transcript_length": len(transcript_text),
                "memories_created": memories_created or [],
                "action_items_created": action_items_created or [],
                "memory_count": len(memories_created) if memories_created else 0,
                "action_item_count": len(action_items_created) if action_items_created else 0,
                "processing_success": processing_success,
                "error_message": error_message,
                "processing_time_ms": processing_time_ms
            }
            
            self.debug_entries.append(entry)
            
            # Keep only the most recent entries
            if len(self.debug_entries) > MAX_DEBUG_ENTRIES:
                self.debug_entries = self.debug_entries[-MAX_DEBUG_ENTRIES:]
            
            # Save to file asynchronously
            asyncio.create_task(self._async_save_to_file())
            
            debug_logger.info(f"Logged memory processing for {user_id}: {len(memories_created or [])} memories, {len(action_items_created or [])} action items")
            debug_logger.info(f"Transcript preview: '{transcript_text[:100]}...'" if len(transcript_text) > 100 else f"Transcript: '{transcript_text}'")
            
        except Exception as e:
            debug_logger.error(f"Error logging memory processing: {e}")
    
    async def _async_save_to_file(self):
        """Async wrapper for saving to file."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._save_to_file)
    
    def get_debug_entries(
        self,
        user_id: Optional[str] = None,
        limit: int = 50,
        since_timestamp: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get debug entries with optional filtering."""
        try:
            entries = self.debug_entries.copy()
            
            # Filter by user_id if provided
            if user_id:
                entries = [e for e in entries if e.get("user_id") == user_id]
            
            # Filter by timestamp if provided
            if since_timestamp:
                entries = [e for e in entries if e.get("unix_timestamp", 0) >= since_timestamp]
            
            # Sort by timestamp (newest first) and limit
            entries.sort(key=lambda x: x.get("unix_timestamp", 0), reverse=True)
            
            return entries[:limit]
            
        except Exception as e:
            debug_logger.error(f"Error getting debug entries: {e}")
            return []
    
    def get_debug_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about memory processing."""
        try:
            entries = self.debug_entries.copy()
            
            # Filter by user_id if provided
            if user_id:
                entries = [e for e in entries if e.get("user_id") == user_id]
            
            if not entries:
                return {
                    "total_entries": 0,
                    "successful_processing": 0,
                    "failed_processing": 0,
                    "total_memories_created": 0,
                    "total_action_items_created": 0,
                    "average_processing_time_ms": 0
                }
            
            successful = [e for e in entries if e.get("processing_success", False)]
            failed = [e for e in entries if not e.get("processing_success", False)]
            
            total_memories = sum(e.get("memory_count", 0) for e in entries)
            total_action_items = sum(e.get("action_item_count", 0) for e in entries)
            
            # Calculate average processing time
            times = [e.get("processing_time_ms", 0) for e in entries if e.get("processing_time_ms")]
            avg_time = sum(times) / len(times) if times else 0
            
            return {
                "total_entries": len(entries),
                "successful_processing": len(successful),
                "failed_processing": len(failed),
                "total_memories_created": total_memories,
                "total_action_items_created": total_action_items,
                "average_processing_time_ms": round(avg_time, 2),
                "success_rate": round(len(successful) / len(entries) * 100, 1) if entries else 0
            }
            
        except Exception as e:
            debug_logger.error(f"Error getting debug stats: {e}")
            return {}

# Global debug instance
memory_debug = MemoryProcessingDebugger() 