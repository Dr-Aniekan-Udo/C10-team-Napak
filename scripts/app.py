"""Thin Gradio front end for the retrieval-grounded Agro-RAG pipeline.

Importing this module only defines UI constants and callbacks.  The configured
pipeline is built from the main guard, or explicitly injected into
``create_app`` for tests and alternate deployments.
"""

from __future__ import annotations

import html
import inspect
import re
import sys
import uuid
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import gradio as gr

# ``python scripts/app.py`` places ``scripts/`` rather than the repository root
# on sys.path. Keep direct launch and ``import scripts.app`` equally usable.
if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.utilities.agent import build_agent
from scripts.utilities.config import AppConfig
from scripts.utilities.document_processor import DocumentProcessor
from scripts.utilities.models import Session
from scripts.utilities.ranker import DEFAULT_MODEL_LOAD_TIMEOUT, build_ranker
from scripts.utilities.sessions import SessionStore, SessionSummary
from scripts.utilities.stream import RagPipeline


APP_TITLE = "AGRO RAG / FIELD NOTES"
SAFETY_NOTICE = "Educational research tool - not agronomic advice"
SEARCHING_STATUS = "Searching extension library..."
RERANKER_LOADING_STATUS = "Local reranker loading..."
RERANKER_READY_STATUS = "Local reranker ready."
GENERATING_STATUS = "Generating grounded answer..."
STREAMING_STATUS = "Writing from retrieved sources..."
SAVED_STATUS = "Saved locally"
LIMITATION_STATUS = "Corpus limitation: no retrieved evidence was returned."
RERANKER_UNAVAILABLE_STATUS = "Reranker unavailable; no result generated"
ERROR_STATUS = "Could not complete request. Check connection, then try again."
RERANKER_READY_TIMEOUT = DEFAULT_MODEL_LOAD_TIMEOUT
MAX_QUEUE_SIZE = 4

STARTER_PROMPTS = (
    ("How can I prepare maize for drought?", "How can I prepare a maize field for a dry season?"),
    ("What should I check when maize leaves show spots?", "What should I check first when maize leaves show spots?"),
    (
        "How can I improve soil nutrients before planting beans?",
        "How can I improve soil nutrients before planting beans?",
    ),
)


APP_CSS = r"""
#agro-rag-app {
  --paper: #F6F1E8;
  --ink: #18251D;
  --forest: #20543E;
  --maize: #E7AE45;
  --clay: #B96549;
  --line: #D9D0C1;
  --moss: #DCE7D8;
  --white-paper: #FCFAF5;
  --shadow: 0 18px 45px rgba(24, 37, 29, .08);
  height: 100dvh;
  min-height: 0;
  max-width: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: var(--paper);
  color: var(--ink);
  font-family: "Source Sans 3", "Source Sans Pro", system-ui, sans-serif;
}

#agro-rag-app *,
#agro-rag-app *::before,
#agro-rag-app *::after {
  box-sizing: border-box;
}

#agro-rag-app .agro-shell {
  display: flex;
  flex-direction: column;
  width: min(100%, 1440px);
  height: 100%;
  flex: 1 1 auto;
  min-height: 0;
  max-width: 100%;
  margin: 0 auto;
  padding: 18px clamp(16px, 3vw, 44px) 24px;
  overflow: hidden;
}

#agro-rag-app #agro-header {
  align-items: center;
  flex: 0 0 auto;
  gap: 18px;
  min-width: 0;
  margin-bottom: 16px;
  padding: 10px 0 14px;
  border-bottom: 1px solid var(--line);
}

#agro-rag-app #brand-lockup,
#agro-rag-app #workspace-meta,
#agro-rag-app #corpus-count,
#agro-rag-app #local-status,
#agro-rag-app #safety-notice {
  border: 0;
  background: transparent;
  padding: 0;
}

#agro-rag-app #brand-lockup {
  flex: 1 1 auto;
}

#agro-rag-app .brand-mark {
  margin: 0;
  color: var(--forest);
  font-family: "Fraunces", Georgia, serif;
  font-size: clamp(1.45rem, 2.2vw, 2.15rem);
  font-weight: 650;
  letter-spacing: -.045em;
  line-height: 1;
}

#agro-rag-app .brand-kicker {
  display: block;
  margin-top: 7px;
  color: var(--clay);
  font-size: .67rem;
  font-weight: 750;
  letter-spacing: .16em;
  text-transform: uppercase;
}

#agro-rag-app #safety-notice {
  flex: 0 1 250px;
  min-width: 0;
  color: rgba(24, 37, 29, .58);
  font-size: .7rem;
  line-height: 1.35;
  text-align: right;
}

#agro-rag-app #workspace-meta {
  display: grid;
  gap: 0;
  margin: 0 0 15px;
  padding: 10px 12px 9px;
  border: 1px solid var(--line);
  border-radius: 14px;
  background: rgba(220, 231, 216, .3);
}

#agro-rag-app #workspace-meta::before {
  margin-bottom: 6px;
  color: var(--clay);
  content: "Workspace status";
  font-size: .62rem;
  font-weight: 800;
  letter-spacing: .13em;
  text-transform: uppercase;
}

#agro-rag-app #workspace-meta #corpus-count,
#agro-rag-app #workspace-meta #local-status {
  min-width: 0;
}

#agro-rag-app .header-stat,
#agro-rag-app .local-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-height: 28px;
  padding: 5px 0;
  border: 0;
  border-bottom: 1px solid rgba(217, 208, 193, .75);
  border-radius: 0;
  color: var(--forest);
  background: transparent;
  font-size: .73rem;
  font-weight: 650;
  line-height: 1.25;
  white-space: normal;
}

#agro-rag-app #workspace-meta #local-status .local-badge {
  border-bottom: 0;
}

#agro-rag-app .local-badge::before {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--forest);
  content: "";
}

#agro-rag-app #app-layout {
  display: grid !important;
  align-items: stretch;
  grid-template-columns: minmax(220px, 17rem) minmax(0, 1fr) minmax(260px, 22rem);
  gap: 18px;
  flex: 1 1 0;
  min-width: 0;
  min-height: 0;
  overflow: hidden;
}

#agro-rag-app .agro-rail,
#agro-rag-app #chat-panel {
  height: 100%;
  min-width: 0;
  min-height: 0;
  border: 1px solid var(--line);
  border-radius: 22px;
  background: rgba(252, 250, 245, .48);
}

#agro-rag-app .agro-rail,
#agro-rag-app #session-rail,
#agro-rag-app #evidence-rail {
  overflow-x: hidden;
  overflow-y: auto;
  scrollbar-color: rgba(32, 84, 62, .42) transparent;
  scrollbar-width: thin;
  box-shadow: 0 8px 24px rgba(24, 37, 29, .035);
}

#agro-rag-app #app-layout > .session-column,
#agro-rag-app #app-layout > .chat-column,
#agro-rag-app #app-layout > .evidence-column {
  width: auto !important;
  min-width: 0 !important;
  min-height: 0;
}

#agro-rag-app #app-layout > .chat-column {
  overflow: hidden;
}

#agro-rag-app #chat-column {
  height: 100%;
}

#agro-rag-app #chat-panel {
  display: flex;
  flex-direction: column;
  overflow: hidden;
  padding: clamp(18px, 2.4vw, 30px);
  background: var(--white-paper);
  box-shadow: var(--shadow);
}

#agro-rag-app .rail-accordion,
#agro-rag-app .rail-accordion > .label-wrap {
  min-height: 0;
  border: 0;
  background: transparent;
}

#agro-rag-app .rail-accordion > .label-wrap {
  padding: 18px 18px 13px;
  color: var(--forest);
  font-family: "Fraunces", Georgia, serif;
  font-size: 1.08rem;
  font-weight: 620;
}

#agro-rag-app .rail-content {
  min-height: 0;
  padding: 0 18px 18px;
}

#agro-rag-app .rail-intro {
  margin: 0 0 16px;
  color: rgba(24, 37, 29, .65);
  font-size: .78rem;
  line-height: 1.45;
}

#agro-rag-app #new-conversation,
#agro-rag-app #load-session,
#agro-rag-app #clear-conversation,
#agro-rag-app #ask-button,
#agro-rag-app .starter-button {
  min-height: 44px;
  border-radius: 12px;
  font-family: "Source Sans 3", "Source Sans Pro", system-ui, sans-serif;
  font-weight: 700;
}

#agro-rag-app #new-conversation {
  width: 100%;
  margin-bottom: 14px;
  border: 1px solid var(--forest);
  background: var(--forest);
}

#agro-rag-app #load-session {
  width: 100%;
  margin-top: 10px;
  border: 1px solid var(--line);
  color: var(--forest);
  background: transparent;
}

#agro-rag-app #clear-conversation {
  width: 100%;
  margin-top: 22px;
  border: 1px solid rgba(185, 101, 73, .35);
  color: var(--clay);
  background: rgba(185, 101, 73, .06);
}

#agro-rag-app #session-groups {
  margin: 14px 0 10px;
  padding-top: 12px;
  border-top: 1px solid var(--line);
}

#agro-rag-app .session-group {
  margin: 0 0 12px;
}

#agro-rag-app .session-group:last-child {
  margin-bottom: 0;
}

#agro-rag-app .session-group h3 {
  margin: 0 0 5px;
  color: var(--clay);
  font-size: .66rem;
  font-weight: 800;
  letter-spacing: .14em;
  text-transform: uppercase;
}

#agro-rag-app .session-group p {
  margin: 0;
  color: rgba(24, 37, 29, .74);
  font-size: .78rem;
  line-height: 1.35;
}

#agro-rag-app #session-list {
  margin-top: 10px;
}

#agro-rag-app #session-list label,
#agro-rag-app #session-list .wrap {
  background: transparent;
}

#agro-rag-app #session-list .wrap {
  gap: 8px;
}

#agro-rag-app .chat-eyebrow {
  margin-bottom: 12px;
  color: var(--clay);
  font-size: .7rem;
  font-weight: 800;
  letter-spacing: .16em;
  text-transform: uppercase;
}

#agro-rag-app #empty-state {
  margin: 4px 0 22px;
  padding: clamp(20px, 3vw, 30px);
  border: 1px solid var(--line);
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(220, 231, 216, .78), rgba(246, 241, 232, .74));
}

#agro-rag-app .empty-title {
  margin: 0;
  color: var(--forest);
  font-family: "Fraunces", Georgia, serif;
  font-size: clamp(1.65rem, 3vw, 2.7rem);
  font-weight: 560;
  letter-spacing: -.045em;
  line-height: 1.05;
}

#agro-rag-app .empty-copy {
  max-width: 54ch;
  margin: 12px 0 22px;
  color: rgba(24, 37, 29, .72);
  font-size: .94rem;
  line-height: 1.55;
}

#agro-rag-app .starter-label {
  margin: 0 0 9px;
  color: var(--ink);
  font-size: .72rem;
  font-weight: 800;
  letter-spacing: .09em;
  text-transform: uppercase;
}

#agro-rag-app .starter-button {
  min-width: 0;
  border: 1px solid rgba(32, 84, 62, .25);
  color: var(--forest);
  background: rgba(252, 250, 245, .62);
  font-size: .78rem;
  text-align: left;
}

#agro-rag-app #chatbot {
  flex: 1 1 0;
  height: auto !important;
  min-width: 0;
  min-height: 0;
  max-height: none;
  border: 0;
  background: transparent;
  overflow-x: hidden;
  overflow-y: auto;
}

#agro-rag-app #chatbot > .wrap,
#agro-rag-app #chatbot .message-wrap {
  min-width: 0;
  min-height: 0;
}

#agro-rag-app #chatbot .wrap {
  overflow-y: auto;
  overflow-x: hidden;
}

#agro-rag-app #chatbot .message-content {
  min-width: 0;
  max-width: 100%;
  overflow-x: auto;
  overflow-y: hidden;
  overflow-wrap: anywhere;
  scrollbar-color: rgba(32, 84, 62, .42) transparent;
  scrollbar-width: thin;
}

#agro-rag-app #chatbot .message-content table {
  width: max-content;
  min-width: 42rem;
  max-width: none;
  table-layout: auto;
}

#agro-rag-app #chatbot .message-content th,
#agro-rag-app #chatbot .message-content td {
  min-width: 8rem;
  white-space: normal;
  word-break: normal;
  overflow-wrap: break-word;
}

#agro-rag-app #chatbot .message-content th:first-child,
#agro-rag-app #chatbot .message-content td:first-child {
  min-width: 11rem;
}

#agro-rag-app #chatbot .message {
  min-width: 0;
  max-width: 72ch;
}

#agro-rag-app #composer {
  align-items: flex-end;
  flex: 0 0 auto;
  gap: 10px;
  margin-top: 20px;
  padding-top: 16px;
  border-top: 1px solid var(--line);
}

#agro-rag-app #message-input {
  min-width: 0;
}

#agro-rag-app #message-input textarea {
  min-height: 88px;
  border: 1px solid var(--line);
  border-radius: 14px;
  color: var(--ink);
  background: var(--paper);
  font-size: .98rem;
  line-height: 1.45;
}

#agro-rag-app #message-input textarea::placeholder {
  color: rgba(24, 37, 29, .48);
}

#agro-rag-app #ask-button {
  min-width: 104px;
  border: 1px solid var(--forest);
  background: var(--forest);
}

#agro-rag-app #query-status {
  flex: 0 0 auto;
  min-height: 24px;
  margin-top: 10px;
  color: rgba(24, 37, 29, .64);
  font-size: .78rem;
}

#agro-rag-app #query-status p {
  margin: 0;
}

#agro-rag-app #evidence-cards {
  min-width: 0;
  min-height: 0;
  border: 0;
  background: transparent;
}

#agro-rag-app .evidence-intro {
  margin: 0 0 15px;
  color: rgba(24, 37, 29, .64);
  font-size: .78rem;
  line-height: 1.45;
}

#agro-rag-app .evidence-empty {
  padding: 18px 0;
  color: rgba(24, 37, 29, .64);
  font-size: .86rem;
  line-height: 1.5;
}

#agro-rag-app .evidence-empty strong {
  display: block;
  margin-bottom: 5px;
  color: var(--forest);
  font-family: "Fraunces", Georgia, serif;
  font-size: 1.12rem;
  font-weight: 600;
}

#agro-rag-app .source-card {
  min-width: 0;
  margin: 0 0 12px;
  padding: 15px;
  border: 1px solid var(--line);
  border-radius: 15px;
  background: rgba(252, 250, 245, .72);
  overflow: visible;
}

#agro-rag-app .source-card:last-child {
  margin-bottom: 0;
}

#agro-rag-app .source-rank {
  display: inline-block;
  margin-bottom: 8px;
  color: var(--clay);
  font-size: .64rem;
  font-weight: 800;
  letter-spacing: .12em;
  text-transform: uppercase;
}

#agro-rag-app .source-title {
  margin: 0;
  color: var(--forest);
  font-family: "Fraunces", Georgia, serif;
  font-size: 1.02rem;
  font-weight: 600;
  line-height: 1.22;
}

#agro-rag-app .source-id,
#agro-rag-app .source-publisher,
#agro-rag-app .source-details {
  color: rgba(24, 37, 29, .68);
  font-size: .73rem;
  line-height: 1.45;
}

#agro-rag-app .source-id {
  margin-top: 6px;
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
  font-size: .67rem;
  overflow-wrap: anywhere;
}

#agro-rag-app .source-publisher {
  margin: 9px 0 0;
}

#agro-rag-app .source-metadata {
  display: grid;
  gap: 6px;
  margin: 11px 0 0;
  padding: 0;
}

#agro-rag-app .source-meta-row {
  display: grid;
  grid-template-columns: minmax(4.7rem, 5.4rem) minmax(0, 1fr);
  gap: 8px;
  align-items: baseline;
  padding-bottom: 6px;
  border-bottom: 1px solid rgba(217, 208, 193, .62);
}

#agro-rag-app .source-meta-row:last-child {
  padding-bottom: 0;
  border-bottom: 0;
}

#agro-rag-app .source-metadata dt {
  color: rgba(24, 37, 29, .5);
  font-size: .62rem;
  font-weight: 800;
  letter-spacing: .08em;
  text-transform: uppercase;
}

#agro-rag-app .source-metadata dd {
  min-width: 0;
  margin: 0;
  color: var(--forest);
  font-size: .73rem;
  overflow-wrap: anywhere;
}

#agro-rag-app .source-details {
  margin-top: 12px;
  padding-top: 10px;
  border-top: 1px solid rgba(217, 208, 193, .75);
}

#agro-rag-app .source-details summary {
  cursor: pointer;
  color: var(--forest);
  font-weight: 750;
}

#agro-rag-app .source-detail-row {
  margin-top: 8px;
}

#agro-rag-app .source-detail-label {
  display: block;
  margin-bottom: 2px;
  color: rgba(24, 37, 29, .5);
  font-size: .64rem;
  font-weight: 800;
  letter-spacing: .08em;
  text-transform: uppercase;
}

#agro-rag-app .source-details a {
  overflow-wrap: anywhere;
  color: var(--forest);
  text-decoration-thickness: 1px;
  text-underline-offset: 3px;
}

#agro-rag-app :is(button, textarea, input, select, summary, [tabindex]):focus-visible {
  outline: 3px solid var(--maize) !important;
  outline-offset: 3px;
}

#agro-rag-app button:hover {
  filter: saturate(1.08) brightness(.98);
}

@media (max-width: 1099px) {
  #agro-rag-app {
    height: auto;
    min-height: 100dvh;
    overflow-x: hidden;
    overflow-y: visible;
  }

  #agro-rag-app .agro-shell {
    height: auto;
    min-height: 100dvh;
    padding: 14px 16px 30px;
    overflow: visible;
  }

  #agro-rag-app #agro-header {
    align-items: flex-start;
    flex-wrap: wrap;
    margin-bottom: 14px;
  }

  #agro-rag-app #brand-lockup {
    flex-basis: 100%;
  }

  #agro-rag-app #safety-notice {
    flex-basis: 100%;
    text-align: left;
  }

  #agro-rag-app #app-layout {
    display: flex !important;
    flex-direction: column;
    flex: 0 0 auto;
    gap: 12px;
    overflow: visible;
  }

  #agro-rag-app .session-column,
  #agro-rag-app .chat-column,
  #agro-rag-app .evidence-column {
    height: auto;
    width: 100% !important;
    min-width: 0 !important;
    flex: 0 0 auto !important;
  }

  #agro-rag-app #app-layout > .chat-column {
    overflow: visible;
  }

  #agro-rag-app #chat-column {
    height: auto;
  }

  #agro-rag-app .chat-column {
    order: 1;
  }

  #agro-rag-app .session-column {
    order: 2;
  }

  #agro-rag-app .evidence-column {
    order: 3;
  }

  #agro-rag-app #session-rail,
  #agro-rag-app #evidence-rail {
    height: auto;
    background: rgba(252, 250, 245, .35);
  }

  #agro-rag-app #session-rail {
    max-height: none;
    overflow: visible;
  }

  #agro-rag-app #evidence-rail {
    max-height: min(32rem, 60vh);
    overflow-x: hidden;
    overflow-y: auto;
  }

  #agro-rag-app #chat-panel {
    height: auto;
    min-height: 0;
    padding: 18px 14px;
    overflow: visible;
  }

  #agro-rag-app #chatbot {
    flex: 0 1 auto;
    height: clamp(18rem, 52vh, 32rem) !important;
    min-height: 18rem;
    max-height: 32rem;
    overflow: hidden;
  }

  #agro-rag-app #chatbot > .wrap {
    overflow-y: auto;
  }

  #agro-rag-app #chat-panel {
    box-shadow: 0 12px 32px rgba(24, 37, 29, .07);
  }

  #agro-rag-app #composer {
    position: sticky;
    bottom: 0;
    z-index: 2;
    margin-right: -14px;
    margin-left: -14px;
    padding: 14px;
    background: rgba(252, 250, 245, .96);
    box-shadow: 0 -10px 24px rgba(24, 37, 29, .06);
  }

  #agro-rag-app #empty-state {
    padding: 20px 16px;
  }

  #agro-rag-app .starter-row {
    flex-direction: column;
  }

  #agro-rag-app .starter-button {
    width: 100%;
  }
}

@media (max-width: 640px) {
  #agro-rag-app .agro-shell {
    padding: 12px 10px 24px;
  }

  #agro-rag-app #agro-header {
    gap: 10px;
  }

  #agro-rag-app #safety-notice {
    max-width: 31ch;
  }

  #agro-rag-app #chat-panel {
    padding: 16px 12px;
  }

  #agro-rag-app #chatbot {
    height: clamp(17rem, 50vh, 28rem) !important;
    min-height: 17rem;
    max-height: 28rem;
  }

  #agro-rag-app #composer {
    margin-right: -12px;
    margin-left: -12px;
    padding: 12px;
  }

  #agro-rag-app #message-input textarea {
    min-height: 78px;
  }

  #agro-rag-app .source-card {
    padding: 13px;
  }

  #agro-rag-app .source-meta-row {
    grid-template-columns: minmax(4.4rem, 4.9rem) minmax(0, 1fr);
    gap: 6px;
  }

  #agro-rag-app .rail-accordion > .label-wrap {
    padding-right: 14px;
    padding-left: 14px;
  }

  #agro-rag-app .rail-content {
    padding-right: 14px;
    padding-left: 14px;
  }

  #agro-rag-app .workspace-meta {
    margin-bottom: 12px;
  }

  #agro-rag-app .brand-mark {
    font-size: 1.35rem;
  }

  #agro-rag-app .brand-kicker {
    font-size: .6rem;
  }

  #agro-rag-app .header-stat,
  #agro-rag-app .local-badge {
    min-height: 26px;
    padding: 4px 0;
    font-size: .7rem;
  }

  #agro-rag-app #composer {
    align-items: stretch;
    flex-direction: column;
  }

  #agro-rag-app #ask-button {
    width: 100%;
  }
}

@media (prefers-reduced-motion: reduce) {
  #agro-rag-app *,
  #agro-rag-app *::before,
  #agro-rag-app *::after {
    scroll-behavior: auto !important;
    transition-duration: .01ms !important;
    animation-duration: .01ms !important;
    animation-iteration-count: 1 !important;
  }
}
"""


@dataclass(frozen=True)
class _SavedSession:
    session_id: str
    date_label: str
    question: str
    modified_at: float


def build_default_pipeline() -> RagPipeline:
    """Build configured dependencies without launching or making provider calls."""

    config = AppConfig.from_env()
    processor = DocumentProcessor.from_csv(config.data_dir / "documents.csv")
    ranker = build_ranker(config, processor)
    agent = build_agent(config)
    sessions = SessionStore(config.session_dir)
    return RagPipeline(processor, ranker, agent, sessions, top_k=config.top_k)


def _shutdown_pipeline(pipeline: object) -> None:
    """Best-effort teardown for optional pipeline/reranker lifecycle owners."""

    target = getattr(pipeline, "ranker", None)
    shutdown = getattr(target, "shutdown", None)
    if not callable(shutdown):
        shutdown = getattr(pipeline, "shutdown", None)
    if not callable(shutdown):
        return
    try:
        shutdown()
    except BaseException:
        # Application shutdown must not replace the original launch failure.
        return


def _launch_default_app() -> None:
    """Retain configured runtime so its owned worker is closed on exit."""

    pipeline = build_default_pipeline()
    try:
        create_app(pipeline).launch()
    finally:
        _shutdown_pipeline(pipeline)


def create_app(pipeline: RagPipeline | None = None) -> gr.Blocks:
    """Compose the offline-safe Gradio UI around an injected or configured pipeline."""

    runtime = pipeline if pipeline is not None else build_default_pipeline()
    metadata = _metadata_for_pipeline(runtime)
    saved_sessions = _list_saved_sessions(runtime)
    initial_session_choices = _radio_choices(saved_sessions)
    initial_query_status = _initial_query_status(runtime)
    reranker_status_timer: gr.Timer | None = None
    corpus_status_html = (
        f'<span class="header-stat">Local corpus · {_corpus_count(runtime)} documents</span>'
    )
    saved_status_html = f'<span class="local-badge">{html.escape(SAVED_STATUS)}</span>'

    blocks_kwargs: dict[str, object] = {
        "analytics_enabled": False,
        "title": APP_TITLE,
        "fill_height": True,
        "fill_width": True,
        "elem_id": "agro-rag-app",
    }
    try:
        blocks_accepts_css = "css" in inspect.signature(gr.Blocks.__init__).parameters
    except (TypeError, ValueError):
        blocks_accepts_css = False
    if blocks_accepts_css:
        blocks_kwargs["css"] = APP_CSS

    with gr.Blocks(**blocks_kwargs) as app:
        with gr.Column(elem_classes=["agro-shell"]):
            with gr.Row(elem_id="agro-header", elem_classes=["agro-header"]):
                gr.HTML(
                    '<p class="brand-mark">AGRO RAG <span>/ FIELD NOTES</span></p>'
                    '<span class="brand-kicker">Agricultural extension library</span>',
                    elem_id="brand-lockup",
                    container=False,
                )
                gr.HTML(
                    f'<span>{html.escape(SAFETY_NOTICE)}</span>',
                    elem_id="safety-notice",
                    container=False,
                )

            with gr.Row(elem_id="app-layout", variant="panel"):
                with gr.Column(
                    scale=4,
                    min_width=220,
                    elem_id="session-rail",
                    elem_classes=["agro-rail", "session-column"],
                ):
                    with gr.Accordion(
                        "Session desk",
                        open=True,
                        elem_id="session-accordion",
                        elem_classes=["rail-accordion"],
                    ):
                        with gr.Column(elem_classes=["rail-content"]):
                            gr.Markdown(
                                "Keep field notes close. Sessions stay on this machine.",
                                elem_classes=["rail-intro"],
                            )
                            with gr.Column(elem_id="workspace-meta", elem_classes=["workspace-meta"]):
                                gr.HTML(
                                    corpus_status_html,
                                    elem_id="corpus-count",
                                    container=False,
                                )
                                gr.HTML(
                                    saved_status_html,
                                    elem_id="local-status",
                                    container=False,
                                )
                            new_conversation = gr.Button(
                                "New conversation",
                                variant="primary",
                                elem_id="new-conversation",
                            )
                            session_groups = gr.HTML(
                                _render_session_groups(saved_sessions),
                                elem_id="session-groups",
                                container=False,
                            )
                            session_list = gr.Radio(
                                choices=initial_session_choices,
                                label="Saved field notes",
                                info="Choose one, then load it.",
                                show_label=True,
                                elem_id="session-list",
                            )
                            load_session = gr.Button(
                                "Load",
                                variant="secondary",
                                elem_id="load-session",
                            )
                            clear_conversation = gr.Button(
                                "Clear current conversation",
                                variant="stop",
                                elem_id="clear-conversation",
                            )

                with gr.Column(
                    scale=11,
                    min_width=500,
                    elem_id="chat-column",
                    elem_classes=["chat-column"],
                ):
                    with gr.Column(elem_id="chat-panel"):
                        gr.HTML(
                            "<div class='chat-eyebrow'>Research workspace / grounded answers</div>",
                            container=False,
                        )
                        empty_state = gr.Column(
                            elem_id="empty-state",
                            elem_classes=["empty-state"],
                        )
                        with empty_state:
                            gr.HTML(
                                '<h1 class="empty-title">Ask the extension library.</h1>'
                                '<p class="empty-copy">Bring one practical field question. '
                                "Answers stay tied to retrieved documents, with the trail visible "
                                "beside the conversation.</p>",
                                container=False,
                            )
                            gr.HTML('<p class="starter-label">Start with a field note</p>', container=False)
                            with gr.Row(elem_id="starter-row", elem_classes=["starter-row"]):
                                starter_buttons: list[gr.Button] = []
                                for index, (label, _prompt) in enumerate(STARTER_PROMPTS, 1):
                                    starter_buttons.append(
                                        gr.Button(
                                            label,
                                            variant="secondary",
                                            size="md",
                                            elem_id=f"starter-prompt-{index}",
                                            elem_classes=["starter-button"],
                                        )
                                    )

                        chatbot = gr.Chatbot(
                            value=[],
                            show_label=False,
                            height="100%",
                            min_height=0,
                            autoscroll=True,
                            sanitize_html=True,
                            render_markdown=True,
                            line_breaks=True,
                            buttons=["copy_all"],
                            placeholder="Your grounded field notes will appear here.",
                            elem_id="chatbot",
                        )
                        with gr.Row(elem_id="composer"):
                            message_input = gr.Textbox(
                                label="Question",
                                placeholder="Ask about crop health, soil, pests, nutrients, or climate...",
                                lines=3,
                                max_lines=8,
                                show_label=False,
                                scale=5,
                                elem_id="message-input",
                            )
                            ask_button = gr.Button(
                                "Ask",
                                variant="primary",
                                size="lg",
                                scale=1,
                                elem_id="ask-button",
                            )
                        query_status = gr.Markdown(
                            initial_query_status,
                            elem_id="query-status",
                        )
                        if _has_reranker_lifecycle(runtime):
                            reranker_status_timer = gr.Timer(
                                value=1.0,
                                active=initial_query_status == RERANKER_LOADING_STATUS,
                            )

                with gr.Column(
                    scale=5,
                    min_width=260,
                    elem_id="evidence-rail",
                    elem_classes=["agro-rail", "evidence-column"],
                ):
                    with gr.Accordion(
                        "Retrieved evidence",
                        open=True,
                        elem_id="evidence-accordion",
                        elem_classes=["rail-accordion"],
                    ):
                        with gr.Column(elem_classes=["rail-content"]):
                            gr.HTML(
                                "<p class='evidence-intro'>Sources that shaped current answer. "
                                "Rank shows retrieval order, not confidence.</p>",
                                container=False,
                            )
                            evidence_cards = gr.HTML(
                                render_source_cards(()),
                                elem_id="evidence-cards",
                                container=False,
                            )

            conversation_state = gr.State(None)
            sources_state = gr.State([])

        message_handler = _make_message_handler(runtime, metadata)
        message_outputs = [
            chatbot,
            message_input,
            ask_button,
            query_status,
            evidence_cards,
            sources_state,
            empty_state,
            conversation_state,
            session_groups,
            session_list,
        ]
        ask_button.click(
            message_handler,
            inputs=[message_input, conversation_state, chatbot, sources_state],
            outputs=message_outputs,
            show_progress="hidden",
            api_name="ask",
        )
        message_input.submit(
            message_handler,
            inputs=[message_input, conversation_state, chatbot, sources_state],
            outputs=message_outputs,
            show_progress="hidden",
            api_name="ask_from_textbox",
        )

        for (_label, prompt), starter_button in zip(STARTER_PROMPTS, starter_buttons):
            starter_button.click(
                lambda value=prompt: value,
                inputs=None,
                outputs=message_input,
                show_progress="hidden",
                api_name=False,
            )

        reset_outputs = [
            conversation_state,
            chatbot,
            evidence_cards,
            sources_state,
            query_status,
            empty_state,
            message_input,
            session_groups,
            session_list,
        ]
        new_conversation.click(
            lambda current_id: _reset_conversation(runtime, current_id, clear=False),
            inputs=conversation_state,
            outputs=reset_outputs,
            show_progress="hidden",
            api_name="new_conversation",
        )
        clear_conversation.click(
            lambda current_id: _reset_conversation(runtime, current_id, clear=True),
            inputs=conversation_state,
            outputs=reset_outputs,
            show_progress="hidden",
            api_name="clear_conversation",
        )
        load_session.click(
            lambda selected_id: _load_conversation(runtime, selected_id, metadata),
            inputs=session_list,
            outputs=reset_outputs,
            show_progress="hidden",
            api_name="load_conversation",
        )
        if reranker_status_timer is not None:
            _register_reranker_status_poll(
                reranker_status_timer,
                runtime,
                query_status,
            )

    _configure_queue(app)
    _start_reranker_warmup(runtime)

    app.show_error = False
    if not blocks_accepts_css:
        # Gradio 6 moved custom CSS from Blocks construction to launch().
        # Keep it on the returned app so the main-guard launch applies it and
        # factory callers can still inspect the complete app configuration.
        app.css = APP_CSS
        app._deprecated_css = APP_CSS  # type: ignore[attr-defined]

    teardown_called = False

    def teardown() -> None:
        nonlocal teardown_called
        if teardown_called:
            return
        teardown_called = True
        _shutdown_pipeline(runtime)

    app.teardown = teardown  # type: ignore[attr-defined]
    return app


def render_source_cards(
    sources: Sequence[object],
    *,
    empty_message: str = "Sources appear here after a question is searched.",
    metadata: Mapping[str, Mapping[str, object]] | None = None,
) -> str:
    """Render source metadata as escaped, score-free HTML cards."""

    source_items = tuple(sources)
    if not source_items:
        return (
            '<div class="evidence-empty">'
            "<strong>No retrieved documents yet.</strong>"
            f"{html.escape(empty_message)}"
            "</div>"
        )

    cards: list[str] = []
    for index, source in enumerate(source_items, 1):
        document_id = (
            _display_value(source, "document_id", f"document-{index}", metadata)
            or f"document-{index}"
        )
        title = (
            _display_value(source, "title", "Untitled extension note", metadata)
            or "Untitled extension note"
        )
        publisher = (
            _display_value(source, "source", "Publisher unavailable", metadata)
            or "Publisher unavailable"
        )
        rank = _display_value(source, "rank", index, metadata) or str(index)
        metadata_rows = []
        for field in ("crop", "country", "origin"):
            value = _display_value(source, field, None, metadata)
            if value is not None:
                metadata_rows.append(
                    '<div class="source-meta-row">'
                    f"<dt>{html.escape(field.title())}</dt>"
                    f"<dd>{html.escape(value)}</dd>"
                    "</div>"
                )

        source_url = _display_value(source, "source_url", None, metadata)
        license_value = _display_value(source, "license", None, metadata)
        detail_rows: list[str] = []
        if source_url is not None:
            safe_url = _safe_http_url(source_url)
            if safe_url is not None:
                escaped_url = html.escape(safe_url, quote=True)
                detail_rows.append(
                    '<div class="source-detail-row">'
                    '<span class="source-detail-label">Source URL</span>'
                    f'<a href="{escaped_url}" target="_blank" rel="noopener noreferrer">'
                    f"{html.escape(safe_url)}</a></div>"
                )
            else:
                detail_rows.append(
                    '<div class="source-detail-row"><span class="source-detail-label">'
                    "Source URL</span>Link unavailable</div>"
                )
        if license_value is not None:
            detail_rows.append(
                '<div class="source-detail-row">'
                '<span class="source-detail-label">License</span>'
                f"{html.escape(license_value)}</div>"
            )
        details = ""
        if detail_rows:
            details = (
                '<details class="source-details"><summary>Source details</summary>'
                + "".join(detail_rows)
                + "</details>"
            )
        metadata_html = (
            f'<dl class="source-metadata">{"".join(metadata_rows)}</dl>'
            if metadata_rows
            else ""
        )
        cards.append(
            f'<article class="source-card" id="source-card-{index}">'
            f'<span class="source-rank">Rank {html.escape(rank)}</span>'
            f'<h3 class="source-title">{html.escape(title)}</h3>'
            f'<div class="source-id">Document ID · {html.escape(document_id)}</div>'
            f'<p class="source-publisher">Source · {html.escape(publisher)}</p>'
            f"{metadata_html}{details}</article>"
        )
    return "".join(cards)


def _make_message_handler(
    pipeline: RagPipeline,
    metadata: Mapping[str, Mapping[str, object]],
):
    def handle_message(
        message: object,
        session_id: object,
        history: object,
        current_sources: object,
    ) -> Iterator[tuple[object, ...]]:
        base_history = _normalise_history(history)
        base_sources = _normalise_sources(current_sources)
        if not isinstance(message, str) or not message.strip():
            yield (
                base_history,
                gr.update(value=message if isinstance(message, str) else "", interactive=True),
                gr.update(interactive=True),
                "Write a question to search the extension library.",
                render_source_cards(base_sources, metadata=metadata),
                list(base_sources),
                gr.update(visible=not base_history),
                session_id if isinstance(session_id, str) else None,
                gr.update(),
                gr.update(),
            )
            return

        question = message.strip()
        if _has_reranker_lifecycle(pipeline):
            yield (
                base_history,
                gr.update(value="", interactive=False),
                gr.update(interactive=False),
                RERANKER_LOADING_STATUS,
                render_source_cards(base_sources, metadata=metadata),
                list(base_sources),
                gr.update(visible=False),
                session_id if isinstance(session_id, str) else None,
                gr.update(),
                gr.update(),
            )
            if not _wait_for_reranker(pipeline):
                selected_id = session_id if isinstance(session_id, str) else None
                groups, choices = _session_control_updates(pipeline, selected_id)
                yield (
                    base_history,
                    gr.update(value=question, interactive=True),
                    gr.update(interactive=True),
                    _reranker_failure_status(pipeline),
                    render_source_cards(base_sources, metadata=metadata),
                    list(base_sources),
                    gr.update(visible=not base_history),
                    selected_id,
                    groups,
                    choices,
                )
                return

            yield (
                base_history,
                gr.update(value="", interactive=False),
                gr.update(interactive=False),
                RERANKER_READY_STATUS,
                render_source_cards(base_sources, metadata=metadata),
                list(base_sources),
                gr.update(visible=False),
                session_id if isinstance(session_id, str) else None,
                gr.update(),
                gr.update(),
            )
        yield (
            base_history,
            gr.update(value="", interactive=False),
            gr.update(interactive=False),
            SEARCHING_STATUS,
            render_source_cards(base_sources, metadata=metadata),
            list(base_sources),
            gr.update(visible=False),
            session_id if isinstance(session_id, str) else None,
            gr.update(),
            gr.update(),
        )
        yield (
            base_history,
            gr.update(value="", interactive=False),
            gr.update(interactive=False),
            GENERATING_STATUS,
            render_source_cards(base_sources, metadata=metadata),
            list(base_sources),
            gr.update(visible=False),
            session_id if isinstance(session_id, str) else None,
            gr.update(),
            gr.update(),
        )

        created_session = False
        active_session_id: str | None = None
        answer_parts: list[str] = []
        retrieved_sources: tuple[object, ...] = base_sources
        try:
            active_session_id, created_session = _ensure_session(pipeline, session_id)
            working_history = base_history + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": ""},
            ]
            stream = pipeline.stream_chat(active_session_id, question)
            for fragment, sources in stream:
                if not isinstance(fragment, str):
                    raise ValueError("non-string answer fragment")
                if not fragment:
                    continue
                answer_parts.append(fragment)
                retrieved_sources = _normalise_sources(sources)
                working_history[-1] = {
                    "role": "assistant",
                    "content": "".join(answer_parts),
                }
                yield (
                    working_history.copy(),
                    gr.update(value="", interactive=False),
                    gr.update(interactive=False),
                    STREAMING_STATUS,
                    render_source_cards(retrieved_sources, metadata=metadata),
                    list(retrieved_sources),
                    gr.update(visible=False),
                    active_session_id,
                    gr.update(),
                    gr.update(),
                )
            if not "".join(answer_parts).strip():
                raise ValueError("empty answer")

            complete_history = base_history + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": "".join(answer_parts)},
            ]
            status = SAVED_STATUS if retrieved_sources else LIMITATION_STATUS
            empty_message = (
                "No matching extension documents were returned. Treat this response as "
                "incomplete and verify against trusted local guidance."
            )
            groups, choices = _session_control_updates(pipeline, active_session_id)
            yield (
                complete_history,
                gr.update(value="", interactive=True),
                gr.update(interactive=True),
                status,
                render_source_cards(
                    retrieved_sources,
                    empty_message=empty_message,
                    metadata=metadata,
                ),
                list(retrieved_sources),
                gr.update(visible=False),
                active_session_id,
                groups,
                choices,
            )
        except Exception:
            if created_session and active_session_id is not None:
                try:
                    _clear_session(pipeline, active_session_id)
                except Exception:
                    pass
            selected_id = session_id if isinstance(session_id, str) else None
            groups, choices = _session_control_updates(pipeline, selected_id)
            failure_status = (
                RERANKER_UNAVAILABLE_STATUS
                if _reranker_is_unavailable(pipeline)
                else ERROR_STATUS
            )
            yield (
                base_history,
                gr.update(value=question, interactive=True),
                gr.update(interactive=True),
                failure_status,
                render_source_cards(base_sources, metadata=metadata),
                list(base_sources),
                gr.update(visible=not base_history),
                session_id if isinstance(session_id, str) else None,
                groups,
                choices,
            )

    return handle_message


def _configure_queue(app: gr.Blocks) -> None:
    """Keep local model work bounded when installed Gradio exposes queue controls."""

    try:
        queue_parameters = inspect.signature(app.queue).parameters
    except (TypeError, ValueError):
        return

    queue_kwargs: dict[str, object] = {}
    if "max_size" in queue_parameters:
        queue_kwargs["max_size"] = MAX_QUEUE_SIZE
    if "default_concurrency_limit" in queue_parameters:
        queue_kwargs["default_concurrency_limit"] = 1
    if queue_kwargs:
        app.queue(**queue_kwargs)


def _start_reranker_warmup(pipeline: object) -> None:
    """Start one guarded background warm-up without blocking app construction."""

    ranker = getattr(pipeline, "ranker", None)
    start_warmup = getattr(ranker, "start_warmup", None)
    if not callable(start_warmup):
        return
    try:
        start_warmup()
    except Exception:
        # Query handling turns lifecycle failures into a safe no-answer state.
        return


def _has_reranker_lifecycle(pipeline: object) -> bool:
    ranker = getattr(pipeline, "ranker", None)
    return callable(getattr(ranker, "start_warmup", None)) and callable(
        getattr(ranker, "wait_until_ready", None)
    )


def _poll_reranker_status(
    pipeline: object,
    current_status: object,
) -> tuple[object, object]:
    """Poll lifecycle state without touching model-loading or message work."""

    stop_timer = gr.update(active=False)
    if not _has_reranker_lifecycle(pipeline) or current_status != RERANKER_LOADING_STATUS:
        return gr.skip(), stop_timer

    ranker = getattr(pipeline, "ranker", None)
    try:
        state = getattr(ranker, "state")
    except Exception:
        return RERANKER_UNAVAILABLE_STATUS, stop_timer
    if state in {"not_started", "loading"}:
        return RERANKER_LOADING_STATUS, gr.skip()
    if state == "ready":
        return RERANKER_READY_STATUS, stop_timer
    if state == "unavailable":
        return RERANKER_UNAVAILABLE_STATUS, stop_timer
    return gr.skip(), stop_timer


def _register_reranker_status_poll(
    timer: gr.Timer,
    pipeline: object,
    query_status: gr.Markdown,
) -> None:
    try:
        tick_parameters = inspect.signature(timer.tick).parameters
    except (TypeError, ValueError):
        return

    tick_kwargs: dict[str, object] = {
        "inputs": [query_status],
        "outputs": [query_status, timer],
        "show_progress": "hidden",
        "api_name": False,
    }
    if "queue" in tick_parameters:
        tick_kwargs["queue"] = False
    timer.tick(
        lambda current_status: _poll_reranker_status(pipeline, current_status),
        **tick_kwargs,
    )


def _wait_for_reranker(pipeline: object) -> bool:
    """Wait for the injected local reranker, accepting only its ready state."""

    ranker = getattr(pipeline, "ranker", None)
    wait_until_ready = getattr(ranker, "wait_until_ready", None)
    if not callable(wait_until_ready):
        return True
    try:
        ready = wait_until_ready(RERANKER_READY_TIMEOUT)
        state = getattr(ranker, "state")
    except Exception:
        return False
    return bool(ready) and state == "ready"


def _reranker_failure_status(pipeline: object) -> str:
    ranker = getattr(pipeline, "ranker", None)
    try:
        if getattr(ranker, "state") == "loading":
            return RERANKER_LOADING_STATUS
    except Exception:
        pass
    return RERANKER_UNAVAILABLE_STATUS


def _reranker_is_unavailable(pipeline: object) -> bool:
    if not _has_reranker_lifecycle(pipeline):
        return False
    try:
        return getattr(getattr(pipeline, "ranker"), "state") == "unavailable"
    except Exception:
        return False


def _initial_query_status(pipeline: object) -> str:
    ranker = getattr(pipeline, "ranker", None)
    if ranker is None or not _has_reranker_lifecycle(pipeline):
        return "Ready when you are."
    try:
        state = getattr(ranker, "state")
    except Exception:
        return RERANKER_UNAVAILABLE_STATUS
    if state in {"not_started", "loading"}:
        return RERANKER_LOADING_STATUS
    if state == "ready":
        return RERANKER_READY_STATUS
    return RERANKER_UNAVAILABLE_STATUS


def _reset_conversation(
    pipeline: RagPipeline,
    session_id: object,
    *,
    clear: bool,
) -> tuple[object, ...]:
    current_id = session_id if isinstance(session_id, str) and session_id.strip() else None
    try:
        if clear and current_id is not None:
            _clear_session(pipeline, current_id)
        new_id = _create_session(pipeline)
        saved = _list_saved_sessions(pipeline)
        return (
            new_id,
            [],
            render_source_cards(()),
            [],
            "New conversation ready.",
            gr.update(visible=True),
            gr.update(value="", interactive=True),
            _render_session_groups(saved),
            gr.update(choices=_radio_choices(saved), value=None),
        )
    except Exception:
        return (
            current_id,
            [],
            render_source_cards(()),
            [],
            "Could not start a new conversation. Try again.",
            gr.update(visible=True),
            gr.update(value="", interactive=True),
            gr.update(),
            gr.update(),
        )


def _load_conversation(
    pipeline: RagPipeline,
    selected_id: object,
    metadata: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[object, ...]:
    if not isinstance(selected_id, str) or not selected_id.strip():
        return (
            None,
            [],
            render_source_cards(()),
            [],
            "Choose a saved conversation, then load it.",
            gr.update(visible=True),
            gr.update(value="", interactive=True),
            gr.update(),
            gr.update(),
        )
    try:
        store = _session_store(pipeline)
        if store is None:
            raise ValueError("session persistence unavailable")
        session = store.load(selected_id)
        history, sources = _session_view(session)
        saved = _list_saved_sessions(pipeline)
        status = f"Loaded locally · {len(session.turns)} turn(s)."
        return (
            selected_id,
            history,
            render_source_cards(sources, metadata=metadata),
            list(sources),
            status,
            gr.update(visible=not history),
            gr.update(value="", interactive=True),
            _render_session_groups(saved),
            gr.update(choices=_radio_choices(saved), value=selected_id),
        )
    except Exception:
        return (
            None,
            [],
            render_source_cards(()),
            [],
            "Could not load that conversation. Choose another saved note.",
            gr.update(visible=True),
            gr.update(value="", interactive=True),
            gr.update(),
            gr.update(),
        )


def _ensure_session(pipeline: RagPipeline, session_id: object) -> tuple[str, bool]:
    if isinstance(session_id, str) and session_id.strip():
        return session_id, False
    return _create_session(pipeline), True


def _create_session(pipeline: RagPipeline) -> str:
    store = _session_store(pipeline)
    if store is not None:
        session = store.create()
        session_id = getattr(session, "session_id", None)
        if isinstance(session_id, str) and session_id.strip():
            return session_id
    return str(uuid.uuid4())


def _clear_session(pipeline: RagPipeline, session_id: str) -> None:
    store = _session_store(pipeline)
    if store is not None:
        store.clear(session_id)


def _session_store(pipeline: object) -> Any | None:
    store = getattr(pipeline, "sessions", None)
    return store if store is not None else None


def _session_control_updates(
    pipeline: RagPipeline,
    selected_id: str | None,
) -> tuple[str, dict[str, object]]:
    saved = _list_saved_sessions(pipeline)
    return (
        _render_session_groups(saved),
        gr.update(choices=_radio_choices(saved), value=selected_id),
    )


def _list_saved_sessions(pipeline: object) -> tuple[_SavedSession, ...]:
    store = _session_store(pipeline)
    if store is None or not callable(getattr(store, "list_summaries", None)):
        return ()
    try:
        summaries = store.list_summaries()
    except (OSError, OverflowError, TypeError, ValueError):
        return ()

    records: list[_SavedSession] = []
    for summary in summaries:
        if not isinstance(summary, SessionSummary):
            continue
        try:
            date_label = datetime.fromtimestamp(summary.modified_at).strftime("%d %b %Y")
        except (OSError, OverflowError, TypeError, ValueError):
            continue
        question = _compact_text(summary.question, 72) or "Untitled field note"
        records.append(
            _SavedSession(
                session_id=summary.session_id,
                date_label=date_label,
                question=question,
                modified_at=summary.modified_at,
            )
        )
    return tuple(records)


def _render_session_groups(records: Sequence[_SavedSession]) -> str:
    if not records:
        return '<p class="rail-intro">No saved field notes yet.</p>'
    grouped: dict[str, list[_SavedSession]] = defaultdict(list)
    for record in records:
        grouped[record.date_label].append(record)
    groups: list[str] = []
    for date_label, date_records in grouped.items():
        questions = "".join(f"<p>{html.escape(record.question)}</p>" for record in date_records)
        groups.append(
            '<section class="session-group">'
            f"<h3>{html.escape(date_label)}</h3>{questions}</section>"
        )
    return "".join(groups)


def _radio_choices(records: Sequence[_SavedSession]) -> list[tuple[str, str]]:
    return [(record.question, record.session_id) for record in records]


def _session_view(session: Session) -> tuple[list[dict[str, str]], tuple[object, ...]]:
    history: list[dict[str, str]] = []
    sources: tuple[object, ...] = ()
    for turn in session.turns:
        history.extend(
            [
                {"role": "user", "content": turn.user.content},
                {"role": "assistant", "content": turn.assistant.content},
            ]
        )
        sources = tuple(turn.sources)
    return history, sources


def _normalise_history(value: object) -> list[dict[str, str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    history: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        role = item.get("role")
        content = item.get("content")
        if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
            text_parts: list[str] = []
            for block in content:
                if not isinstance(block, Mapping) or block.get("type") != "text":
                    continue
                text = block.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
            content = "".join(text_parts) if text_parts else None
        if role in {"user", "assistant"} and isinstance(content, str):
            history.append({"role": role, "content": content})
    return history


def _normalise_sources(value: object) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(value)


def _metadata_for_pipeline(pipeline: object) -> dict[str, Mapping[str, object]]:
    metadata: dict[str, Mapping[str, object]] = {}
    candidates = (
        getattr(pipeline, "document_metadata", None),
        getattr(pipeline, "metadata", None),
    )
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            for document_id, values in candidate.items():
                if isinstance(values, Mapping):
                    metadata[str(document_id)] = values
    return metadata


def _display_value(
    source: object,
    field: str,
    fallback: object,
    metadata: Mapping[str, Mapping[str, object]] | None,
) -> str | None:
    value = _field_value(source, field)
    if value is None or (isinstance(value, str) and not value.strip()):
        document_id = _field_value(source, "document_id")
        extra = metadata.get(str(document_id), {}) if metadata and document_id is not None else {}
        value = extra.get(field, fallback)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return str(value).strip()


def _field_value(source: object, field: str) -> object:
    if isinstance(source, Mapping):
        return source.get(field)
    return getattr(source, field, None)


def _safe_http_url(value: str) -> str | None:
    candidate = value.strip()
    try:
        parsed = urlparse(candidate)
        parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or not parsed.hostname
        or any(character.isspace() for character in candidate)
    ):
        return None
    return candidate


def _compact_text(value: str, limit: int) -> str:
    compact = re.sub(r"\s+", " ", value).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _corpus_count(pipeline: object) -> str:
    count = getattr(pipeline, "corpus_count", None)
    if isinstance(count, int) and count >= 0:
        return str(count)
    return "—"


if __name__ == "__main__":
    _launch_default_app()
