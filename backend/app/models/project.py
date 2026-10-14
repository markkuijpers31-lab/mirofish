"""
Project context management
Used to persist project state on the server side, avoiding the frontend passing large amounts of data between interfaces
"""

import os
import re
import json
import uuid
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass, field, asdict
from ..config import Config


class ProjectStatus(str, Enum):
    """Project status"""

    CREATED = "created"  # just created, files uploaded
    ONTOLOGY_GENERATED = "ontology_generated"  # ontology generated
    GRAPH_BUILDING = "graph_building"  # graph building in progress
    GRAPH_COMPLETED = "graph_completed"  # graph build completed
    FAILED = "failed"  # failed


@dataclass
class Project:
    """Project data model"""

    project_id: str
    name: str
    status: ProjectStatus
    created_at: str
    updated_at: str

    # File info
    files: List[Dict[str, str]] = field(
        default_factory=list
    )  # [{filename, path, size}]
    total_text_length: int = 0

    # Ontology info (populated after endpoint 1 generates it)
    ontology: Optional[Dict[str, Any]] = None
    analysis_summary: Optional[str] = None

    # Graph info (populated after endpoint 2 completes)
    graph_id: Optional[str] = None
    graph_build_task_id: Optional[str] = None

    # Config
    simulation_requirement: Optional[str] = None
    # Free-text notes submitted with the upload (part of the assessment brief)
    additional_context: Optional[str] = None
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Error info
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "project_id": self.project_id,
            "name": self.name,
            "status": self.status.value
            if isinstance(self.status, ProjectStatus)
            else self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "files": self.files,
            "total_text_length": self.total_text_length,
            "ontology": self.ontology,
            "analysis_summary": self.analysis_summary,
            "graph_id": self.graph_id,
            "graph_build_task_id": self.graph_build_task_id,
            "simulation_requirement": self.simulation_requirement,
            "additional_context": self.additional_context,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Project":
        """Create from dictionary"""
        status = data.get("status", "created")
        if isinstance(status, str):
            status = ProjectStatus(status)

        return cls(
            project_id=data["project_id"],
            name=data.get("name", "Unnamed Project"),
            status=status,
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            files=data.get("files", []),
            total_text_length=data.get("total_text_length", 0),
            ontology=data.get("ontology"),
            analysis_summary=data.get("analysis_summary"),
            graph_id=data.get("graph_id"),
            graph_build_task_id=data.get("graph_build_task_id"),
            simulation_requirement=data.get("simulation_requirement"),
            additional_context=data.get("additional_context"),
            chunk_size=data.get("chunk_size", 500),
            chunk_overlap=data.get("chunk_overlap", 50),
            error=data.get("error"),
        )


class ProjectManager:
    """Project manager - responsible for project persistence storage and retrieval"""

    # Project storage root directory
    PROJECTS_DIR = os.path.join(Config.UPLOAD_FOLDER, "projects")

    @classmethod
    def _ensure_projects_dir(cls):
        """Ensure the projects directory exists"""
        os.makedirs(cls.PROJECTS_DIR, exist_ok=True)

    @classmethod
    def _get_project_dir(cls, project_id: str) -> str:
        """Get the project directory path"""
        return os.path.join(cls.PROJECTS_DIR, project_id)

    @classmethod
    def _get_project_meta_path(cls, project_id: str) -> str:
        """Get the project metadata file path"""
        return os.path.join(cls._get_project_dir(project_id), "project.json")

    @classmethod
    def _get_project_files_dir(cls, project_id: str) -> str:
        """Get the project file storage directory"""
        return os.path.join(cls._get_project_dir(project_id), "files")

    @classmethod
    def _get_project_text_path(cls, project_id: str) -> str:
        """Get the path where the project's extracted text is stored"""
        return os.path.join(cls._get_project_dir(project_id), "extracted_text.txt")

    @classmethod
    def _get_project_sources_dir(cls, project_id: str) -> str:
        """Get the directory holding per-document extracted text"""
        return os.path.join(cls._get_project_dir(project_id), "sources")

    @classmethod
    def create_project(cls, name: str = "Unnamed Project") -> Project:
        """
        Create a new project

        Args:
            name: Project name

        Returns:
            The newly created Project object
        """
        cls._ensure_projects_dir()

        project_id = f"proj_{uuid.uuid4().hex[:12]}"
        now = datetime.now().isoformat()

        project = Project(
            project_id=project_id,
            name=name,
            status=ProjectStatus.CREATED,
            created_at=now,
            updated_at=now,
        )

        # Create project directory structure
        project_dir = cls._get_project_dir(project_id)
        files_dir = cls._get_project_files_dir(project_id)
        os.makedirs(project_dir, exist_ok=True)
        os.makedirs(files_dir, exist_ok=True)

        # Save project metadata
        cls.save_project(project)

        return project

    @classmethod
    def save_project(cls, project: Project) -> None:
        """Save project metadata"""
        project.updated_at = datetime.now().isoformat()
        meta_path = cls._get_project_meta_path(project.project_id)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(project.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def get_project(cls, project_id: str) -> Optional[Project]:
        """
        Get a project

        Args:
            project_id: Project ID

        Returns:
            Project object, or None if it does not exist
        """
        meta_path = cls._get_project_meta_path(project_id)

        if not os.path.exists(meta_path):
            return None

        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return Project.from_dict(data)

    @classmethod
    def list_projects(cls, limit: int = 50) -> List[Project]:
        """
        List all projects

        Args:
            limit: Return count limit

        Returns:
            List of projects, sorted by creation time in descending order
        """
        cls._ensure_projects_dir()

        projects = []
        for project_id in os.listdir(cls.PROJECTS_DIR):
            project = cls.get_project(project_id)
            if project:
                projects.append(project)

        # Sort by creation time in descending order
        projects.sort(key=lambda p: p.created_at, reverse=True)

        return projects[:limit]

    @classmethod
    def delete_project(cls, project_id: str) -> bool:
        """
        Delete a project and all its files

        Args:
            project_id: Project ID

        Returns:
            Whether the deletion was successful
        """
        project_dir = cls._get_project_dir(project_id)

        if not os.path.exists(project_dir):
            return False

        shutil.rmtree(project_dir)
        return True

    @classmethod
    def save_file_to_project(
        cls, project_id: str, file_storage, original_filename: str
    ) -> Dict[str, Any]:
        """
        Save an uploaded file to the project directory

        Args:
            project_id: Project ID
            file_storage: Flask FileStorage object
            original_filename: Original filename

        Returns:
            File info dictionary {filename, path, size}
        """
        files_dir = cls._get_project_files_dir(project_id)
        os.makedirs(files_dir, exist_ok=True)

        # Generate a safe filename
        ext = os.path.splitext(original_filename)[1].lower()
        safe_filename = f"{uuid.uuid4().hex[:8]}{ext}"
        file_path = os.path.join(files_dir, safe_filename)

        # Save file
        file_storage.save(file_path)

        # Get file size
        file_size = os.path.getsize(file_path)

        return {
            "original_filename": original_filename,
            "saved_filename": safe_filename,
            "path": file_path,
            "size": file_size,
        }

    @classmethod
    def save_extracted_text(cls, project_id: str, text: str) -> None:
        """Save the extracted text"""
        text_path = cls._get_project_text_path(project_id)
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)

    @classmethod
    def get_extracted_text(cls, project_id: str) -> Optional[str]:
        """Get the extracted text"""
        text_path = cls._get_project_text_path(project_id)

        if not os.path.exists(text_path):
            return None

        with open(text_path, "r", encoding="utf-8") as f:
            return f.read()

    @classmethod
    def get_project_files(cls, project_id: str) -> List[str]:
        """Get all file paths of the project"""
        files_dir = cls._get_project_files_dir(project_id)

        if not os.path.exists(files_dir):
            return []

        return [
            os.path.join(files_dir, f)
            for f in os.listdir(files_dir)
            if os.path.isfile(os.path.join(files_dir, f))
        ]

    @classmethod
    def save_source_text(
        cls, project_id: str, index: int, original_filename: str, text: str
    ) -> str:
        """
        Persist the extracted text of a single source document

        Args:
            project_id: Project ID
            index: Position of the document in the upload (1-based)
            original_filename: Original filename as uploaded
            text: Extracted text

        Returns:
            Filename of the stored text, relative to the project's sources/ dir
        """
        sources_dir = cls._get_project_sources_dir(project_id)
        os.makedirs(sources_dir, exist_ok=True)
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.splitext(original_filename)[0])
        text_filename = f"{index:02d}_{stem[:60] or 'document'}.txt"
        with open(os.path.join(sources_dir, text_filename), "w", encoding="utf-8") as f:
            f.write(text or "")
        return text_filename

    @classmethod
    def get_source_documents(cls, project_id: str) -> List[Dict[str, Any]]:
        """
        Get every source document of a project with its extracted text

        New projects store one text file per document (see save_source_text).
        Older projects only have the concatenated extracted_text.txt; it is split
        on its "=== filename ===" headers as a read-only fallback.

        Returns:
            List of {original_filename, saved_filename, original_path, extension,
            size, text, text_length, extraction_status, error, origin}
        """
        project = cls.get_project(project_id)
        if not project:
            return []

        files_dir = cls._get_project_files_dir(project_id)
        sources_dir = cls._get_project_sources_dir(project_id)
        documents: List[Dict[str, Any]] = []
        has_per_file_text = any(f.get("text_file") for f in project.files)

        if has_per_file_text:
            for f in project.files:
                text = ""
                text_file = f.get("text_file")
                if text_file:
                    text_path = os.path.join(sources_dir, text_file)
                    if os.path.exists(text_path):
                        with open(text_path, "r", encoding="utf-8") as fh:
                            text = fh.read()
                saved = f.get("saved_filename")
                original_path = os.path.join(files_dir, saved) if saved else None
                documents.append({
                    "original_filename": f.get("filename"),
                    "saved_filename": saved,
                    "original_path": original_path if original_path and os.path.exists(original_path) else None,
                    "extension": f.get("extension") or os.path.splitext(f.get("filename") or "")[1].lower(),
                    "size": f.get("size"),
                    "text": text,
                    "text_length": len(text),
                    "extraction_status": f.get("extraction_status", "ok" if text else "empty"),
                    "error": f.get("error"),
                    "origin": "per_file_text",
                })
            return documents

        # Legacy fallback: split the concatenated text on its headers
        full_text = cls.get_extracted_text(project_id) or ""
        parts = re.split(r"^=== (.+?) ===$", full_text, flags=re.MULTILINE)
        sections: Dict[str, str] = {}
        for i in range(1, len(parts) - 1, 2):
            sections[parts[i].strip()] = parts[i + 1].strip()

        for f in project.files:
            name = f.get("filename")
            text = sections.pop(name, "")
            documents.append({
                "original_filename": name,
                "saved_filename": f.get("saved_filename"),
                "original_path": None,
                "extension": os.path.splitext(name or "")[1].lower(),
                "size": f.get("size"),
                "text": text,
                "text_length": len(text),
                "extraction_status": "ok" if text else "missing",
                "error": None if text else "Text not found in legacy extracted_text.txt",
                "origin": "legacy_extracted_text",
            })
        # Headers without a matching file entry are still evidence that was used
        for name, text in sections.items():
            documents.append({
                "original_filename": name,
                "saved_filename": None,
                "original_path": None,
                "extension": os.path.splitext(name)[1].lower(),
                "size": None,
                "text": text,
                "text_length": len(text),
                "extraction_status": "ok" if text else "empty",
                "error": None,
                "origin": "legacy_extracted_text",
            })
        return documents
