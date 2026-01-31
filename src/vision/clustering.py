"""Face clustering service for grouping unlabeled faces into persons.

This module provides HDBSCAN-based clustering of face embeddings stored in ChromaDB,
creating Person records and updating Face associations in both ChromaDB and SQLite.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from src.db.models.vision import Face, Person, managed_session

logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    """Result from face clustering operation.

    Attributes:
        clusters_created: Number of new person clusters created.
        faces_clustered: Total faces assigned to clusters.
        noise_faces: Faces that couldn't be clustered (outliers).
        new_person_ids: List of newly created person IDs.
    """
    clusters_created: int
    faces_clustered: int
    noise_faces: int
    new_person_ids: list[str]


def cluster_unlabeled_faces(
    min_cluster_size: int = 3,
    min_samples: int = 2,
) -> ClusteringResult:
    """Cluster unlabeled faces into persons using HDBSCAN.

    Retrieves all faces without person_id from ChromaDB, clusters them
    by embedding similarity, and creates Person records for each cluster.

    Args:
        min_cluster_size: Minimum faces to form a cluster (default: 3).
        min_samples: Min samples for core point in HDBSCAN (default: 2).

    Returns:
        ClusteringResult with statistics about the operation.

    Raises:
        ImportError: If hdbscan or numpy are not installed.
        Exception: If database operations fail.
    """
    import hdbscan
    import numpy as np
    from src.db.chroma_client import get_faces_collection

    # Get all unlabeled faces from ChromaDB
    collection = get_faces_collection()
    all_faces = collection.get(
        where={"person_id": {"$exists": False}},
        include=["embeddings", "metadatas"],
    )

    if not all_faces["ids"]:
        logger.info("No unlabeled faces to cluster")
        return ClusteringResult(
            clusters_created=0,
            faces_clustered=0,
            noise_faces=0,
            new_person_ids=[],
        )

    logger.info(f"Clustering {len(all_faces['ids'])} unlabeled faces")

    # Run HDBSCAN clustering
    embeddings = np.array(all_faces["embeddings"])
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(embeddings)

    # Get unique cluster labels (excluding noise label -1)
    unique_labels = set(labels)
    unique_labels.discard(-1)

    new_person_ids: list[str] = []
    faces_clustered = 0
    noise_faces = int(np.sum(labels == -1))

    # Create persons and assign faces
    with managed_session() as session:
        for label in unique_labels:
            person_id = str(uuid.uuid4())
            person = Person(id=person_id)
            session.add(person)
            new_person_ids.append(person_id)

            # Get indices of faces in this cluster
            face_indices = np.where(labels == label)[0]

            for idx in face_indices:
                face_id = all_faces["ids"][idx]

                # Update ChromaDB metadata
                metadata = all_faces["metadatas"][idx] if all_faces["metadatas"] else {}
                metadata["person_id"] = person_id
                collection.update(ids=[face_id], metadatas=[metadata])

                # Update SQLite
                face_record = session.query(Face).filter(Face.id == face_id).first()
                if face_record:
                    face_record.person_id = person_id

                faces_clustered += 1

            person.photo_count = len(face_indices)

    logger.info(
        f"Clustering complete: {len(new_person_ids)} clusters, "
        f"{faces_clustered} faces assigned, {noise_faces} noise"
    )

    return ClusteringResult(
        clusters_created=len(new_person_ids),
        faces_clustered=faces_clustered,
        noise_faces=noise_faces,
        new_person_ids=new_person_ids,
    )
