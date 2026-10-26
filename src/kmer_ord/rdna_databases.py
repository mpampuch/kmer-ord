"""rDNA-miner database names.

This module has no kmer_ord imports. The CLI loads it at startup, so it must
not pull in env_manager (conda helpers, logging) or a half-initialized import
of that module takes down every kmer-ord command, including setup.
"""

RDNA_DB_CHOICES = ("rfam", "silva", "pr2", "all")
