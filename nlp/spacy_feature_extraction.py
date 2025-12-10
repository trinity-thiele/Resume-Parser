import google.generativeai as genai
import json
import time
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

# Get API key from environment variable
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Optional: pdfplumber for PDF extraction
try:
    import pdfplumber
except ImportError:
    print("[WARNING] pdfplumber not installed. Run: pip install pdfplumber")
    pdfplumber = None

# spaCy for NLP-based skill extraction
try:
    import spacy
    from spacy.matcher import PhraseMatcher
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] spaCy not available: {e}")
    print("[WARNING] Install with: pip install spacy && python -m spacy download en_core_web_sm")
    nlp = None
    PhraseMatcher = None
    SPACY_AVAILABLE = False


# ============================================================
# CLUSTER DEFINITIONS 
# ============================================================

CATEGORY_TO_CLUSTER = {
    # Cluster 0: Backend Development
    'Java Developer': 0,
    'Python Developer': 0,
    'DotNet Developer': 0,
    'Backend Developer': 0,
    'SQL Developer': 0,

    # Cluster 1: Frontend Development
    'React Developer': 1,
    'Frontend Developer': 1,
    'Web Designing': 1,
    'UI/UX Designer': 1,

    # Cluster 2: Full Stack Development
    'Full Stack Developer': 2,
    'Software Developer': 2,
    'Technical Lead': 2,

    # Cluster 3: Data Science & AI/ML
    'Data Science': 3,
    'Hadoop': 3,
    'Machine Learning Engineer': 3,
    'AI Engineer': 3,
    'ETL Developer': 3,
    'Business Analyst': 3,

    # Cluster 4: Database Administration
    'Database': 4,
    'Database Administrator': 4,

    # Cluster 5: DevOps & Cloud
    'DevOps': 5,
    'DevOps Engineer': 5,
    'Cloud Engineer': 5,
    'Site Reliability Engineer': 5,

    # Cluster 6: Quality Assurance
    'Testing': 6,
    'Automation Testing': 6,
    'QA Engineer': 6,

    # Cluster 7: Security
    'Network Security Engineer': 7,
    'Cybersecurity Analyst': 7,

    # Cluster 8: Systems Administration
    'System Administrator': 8,

    # Cluster 9: Mobile Development
    'Mobile Developer': 9,

    # Cluster 10: Specialized Technologies
    'Blockchain': 10,
    'Blockchain Developer': 10,
    'SAP Developer': 10,

    # Cluster 11: Engineering Leadership
    'Engineering Manager': 11,
    'Principal Engineer': 11,
    'Product Manager': 11,

    # Cluster 12: Content & Documentation
    'Digital Media': 12,
    'Technical Writer': 12,
}

# Human-readable cluster names for the prompt
CLUSTER_DESCRIPTIONS = {
    0: "Backend Development (Java, Python, .NET, SQL development)",
    1: "Frontend Development (React, Web Design, UI/UX)",
    2: "Full Stack Development (Full Stack, Software Developer, Tech Lead)",
    3: "Data Science & AI/ML (Data Science, ML, AI, Hadoop, ETL, Business Analyst)",
    4: "Database Administration (DBA, Database management)",
    5: "DevOps & Cloud (DevOps, Cloud Engineering, SRE)",
    6: "Quality Assurance (Testing, Automation Testing, QA)",
    7: "Security (Network Security, Cybersecurity)",
    8: "Systems Administration (SysAdmin)",
    9: "Mobile Development (iOS, Android, Mobile apps)",
    10: "Specialized Technologies (Blockchain, SAP)",
    11: "Engineering Leadership (Engineering Manager, Principal Engineer, Product Manager)",
    12: "Content & Documentation (Digital Media, Technical Writing)",
}


# ============================================================
# KNOWN SKILLS DATABASE (from jz_skill_patterns.jsonl)
# ============================================================
# These are validated technical skills used for better matching

KNOWN_SKILLS = {
    # Programming Languages
    ".net", "asp.net", "c", "c#", "c++", "cobol", "f#", "go", "java", "javascript",
    "kotlin", "matlab", "php", "python", "r", "ruby", "rust", "scala", "swift",
    "typescript", "sql", "html", "html5", "css", "xml", "json", "yaml",
    
    # Frontend Frameworks & Libraries
    "react", "react native", "react router", "angular", "angular 2", "angularjs",
    "vue.js", "vue js", "vuex", "vuetify", "vuepress", "preact", "svelte",
    "jquery", "jquery mobile", "bootstrap", "tailwind css", "material ui",
    "semantic ui react", "next.js", "nuxt.js", "gatsby",
    
    # Backend Frameworks
    "node.js", "node js", "express", "express.js", "django", "django rest framework",
    "flask", "fastapi", "spring", "spring boot", "spring cloud", "rails", "rails api",
    "laravel", "asp.net core", "hapi", ".net core",
    
    # Databases
    "mysql", "postgresql", "postgres", "mongodb", "mongodb atlas", "redis",
    "redis cloud", "sqlite", "oracle", "microsoft sql server", "mariadb",
    "cassandra", "couchdb", "dynamodb", "amazon dynamodb", "firebase",
    "cloud firestore", "elasticsearch", "neo4j", "graphql",
    
    # Cloud Platforms & Services
    "aws", "amazon web services", "aws lambda", "aws elastic beanstalk",
    "aws cloudformation", "aws codebuild", "aws codecommit", "aws codedeploy",
    "aws codepipeline", "aws fargate", "aws iam", "aws opsworks",
    "azure", "microsoft azure", "azure functions", "azure cosmos db",
    "azure machine learning", "azure storage", "azure websites",
    "gcp", "google cloud", "google cloud functions", "google cloud storage",
    "google cloud sql", "google cloud pub/sub", "google kubernetes engine",
    "heroku", "digitalocean", "linode", "rackspace", "cloudflare",
    
    # DevOps & CI/CD
    "docker", "docker compose", "docker swarm", "kubernetes", "k8s",
    "jenkins", "gitlab ci", "github actions", "circleci", "travis ci",
    "terraform", "ansible", "puppet", "chef", "vagrant",
    "prometheus", "grafana", "nagios", "datadog", "splunk",
    "nginx", "apache", "load balancing",
    
    # Data Science & ML
    "machine learning", "deep learning", "neural networks", "nlp",
    "natural language processing", "computer vision", "tensorflow",
    "pytorch", "keras", "scikit-learn", "scikit learn", "pandas", "numpy",
    "scipy", "matplotlib", "seaborn", "jupyter", "anaconda",
    "spark", "apache spark", "hadoop", "hive", "pig", "kafka",
    "airflow", "luigi", "dbt", "snowflake", "redshift", "bigquery",
    "tableau", "power bi", "looker", "metabase",
    "random forest", "xgboost", "adaboost", "k-nearest neighbors",
    
    # Testing & QA
    "selenium", "cypress", "jest", "mocha", "chai", "jasmine",
    "pytest", "unittest", "junit", "testng", "cucumber",
    "postman", "insomnia", "soapui", "jmeter", "locust",
    "functional testing", "load testing", "performance testing",
    "unit testing", "integration testing", "regression testing",
    "test automation", "qa automation", "browser testing",
    
    # Security
    "network security", "cybersecurity", "penetration testing",
    "vulnerability assessment", "security audit", "owasp",
    "encryption", "ssl", "tls", "oauth", "jwt", "saml",
    "firewall", "ids", "ips", "siem", "soar",
    
    # Mobile Development
    "android", "android sdk", "android studio", "ios", "xcode",
    "react native", "flutter", "xamarin", "ionic", "cordova",
    "swift", "objective-c", "kotlin", "java android",
    
    # Version Control & Collaboration
    "git", "github", "gitlab", "bitbucket", "svn", "mercurial",
    "jira", "confluence", "trello", "asana", "slack",
    
    # Design Tools
    "figma", "sketch", "adobe photoshop", "adobe illustrator",
    "adobe xd", "invision", "zeplin", "principle",
    
    # APIs & Integration
    "rest", "rest api", "restful", "graphql", "soap", "grpc",
    "webhooks", "api gateway", "swagger", "openapi",
    
    # Blockchain
    "blockchain", "ethereum", "solidity", "web3", "smart contracts",
    "hyperledger", "bitcoin", "cryptocurrency",
    
    # SAP
    "sap", "sap hana", "sap abap", "sap fiori", "sap s/4hana",
    
    # Project Management & Methodologies
    "agile", "scrum", "kanban", "waterfall", "lean",
    "project management", "product management", "sdlc",
    
    # Other Technical Skills
    "linux", "unix", "windows server", "bash", "shell scripting",
    "powershell", "virtualization", "vmware", "hyperv",
    "networking", "tcp/ip", "dns", "dhcp", "vpn",
    "microservices", "serverless", "event-driven architecture",
    "message queues", "rabbitmq", "activemq", "zeromq",
    "caching", "memcached", "varnish", "cdn",
    "elasticsearch", "solr", "lucene",
    "etl", "data pipeline", "data warehouse", "data lake",
    "bi", "business intelligence", "data visualization",
    "technical writing", "documentation", "api documentation",
}


# ============================================================
#  RESUME EXTRACTION 
# ============================================================

# Section heading patterns
SECTION_PATTERNS = {
    "skills": [r"key skills?", r"skills?", r"technical skills?", r"core competencies"],
    "education": [r"education", r"academic background", r"qualifications"],
    "experience": [r"professional experience", r"work experience", r"experience", r"employment history"],
}

ALL_HEADING_PATTERNS = [
    r"key skills?", r"skills?", r"technical skills?", r"core competencies",
    r"education", r"academic background", r"qualifications",
    r"professional experience", r"work experience", r"experience", r"employment history",
    r"certifications?", r"projects?", r"summary", r"profile", r"professional summary",
    r"objective", r"achievements", r"awards", r"publications", r"references",
]


def read_pdf(path: str) -> str:
    """Extract text from PDF using pdfplumber."""
    if pdfplumber is None:
        raise ImportError("pdfplumber is required. Install with: pip install pdfplumber")
    
    parts: List[str] = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                parts.append(text)
    except Exception as e:
        print(f"[ERROR] Failed to read {path}: {e}")
        return ""
    return "\n".join(parts)


def is_heading_line(line: str) -> bool:
    """Return True if the line looks like a section heading."""
    text = line.strip().lower()
    if not text:
        return False
    for pat in ALL_HEADING_PATTERNS:
        if re.fullmatch(pat, text, flags=re.IGNORECASE):
            return True
    return False


def match_section_name(line: str) -> str:
    """If 'line' is a target heading, return the section name."""
    text = line.strip().lower()
    for section_name, patterns in SECTION_PATTERNS.items():
        for pat in patterns:
            if re.fullmatch(pat, text, flags=re.IGNORECASE):
                return section_name
    return ""


def segment_sections_by_headings(text: str) -> Dict[str, List[str]]:
    """Split resume text and extract raw lines for skills, education, experience."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    sections = {"skills": [], "education": [], "experience": []}
    
    i = 0
    n = len(lines)
    
    while i < n:
        line = lines[i]
        section_name = match_section_name(line)
        
        if section_name:
            i += 1
            collected = []
            while i < n:
                next_line = lines[i]
                if is_heading_line(next_line):
                    break
                collected.append(next_line)
                i += 1
            cleaned = [ln for ln in collected if ln.strip()]
            sections[section_name].extend(cleaned)
        else:
            i += 1
    
    return sections


def extract_skills_from_lines(lines: List[str]) -> List[str]:
    """Convert skills section lines into a clean list using spaCy."""
    raw_items = []
    for ln in lines:
        parts = re.split(r"[•\u2022\|\-·]|,", ln)
        for p in parts:
            item = p.strip()
            if len(item) > 1:
                raw_items.append(item)
    
    # Use spaCy for intelligent filtering
    if SPACY_AVAILABLE and nlp is not None:
        filtered = []
        for it in raw_items:
            doc = nlp(it)
            # Filter out items that are full sentences (have subject-verb structure)
            has_verb = any(tok.pos_ == "VERB" for tok in doc)
            has_subject = any(tok.dep_ == "nsubj" for tok in doc)
            
            # Keep if: no verb, or short phrase, or is a known skill
            if (not has_verb) or (not has_subject) or (len(doc) <= 5) or (it.lower() in KNOWN_SKILLS):
                filtered.append(it)
        raw_items = filtered
    
    # Deduplicate
    seen = set()
    skills = []
    for item in raw_items:
        low = item.lower()
        if low not in seen:
            seen.add(low)
            skills.append(item)
    return skills


# ============================================================
# SPACY SKILL MATCHER
# ============================================================

class SkillExtractor:
    """
    Uses spaCy's PhraseMatcher to extract known skills from text.
    More accurate than regex-based matching.
    """
    
    def __init__(self, skill_list: set):
        """
        Initialize the skill extractor with a list of known skills.
        
        Args:
            skill_list: Set of known skill strings
        """
        self.skill_list = skill_list
        self.matcher = None
        self.skill_patterns = {}
        
        if SPACY_AVAILABLE and nlp is not None:
            self._build_matcher()
    
    def _build_matcher(self):
        """Build the spaCy PhraseMatcher with skill patterns."""
        self.matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        
        # Group skills and create patterns
        patterns = []
        for skill in self.skill_list:
            # Create a doc pattern for each skill
            pattern = nlp.make_doc(skill)
            patterns.append(pattern)
            # Store mapping from pattern text to original skill
            self.skill_patterns[skill.lower()] = skill
        
        # Add all patterns to matcher
        self.matcher.add("SKILL", patterns)
        print(f"[INFO] Built spaCy PhraseMatcher with {len(patterns)} skill patterns")
    
    def extract_skills(self, text: str) -> List[str]:
        """
        Extract skills from text using spaCy PhraseMatcher.
        
        Args:
            text: The text to extract skills from
            
        Returns:
            List of found skills (deduplicated)
        """
        if not SPACY_AVAILABLE or self.matcher is None:
            # Fallback to regex if spaCy not available
            return self._regex_fallback(text)
        
        # Process text with spaCy
        doc = nlp(text)
        
        # Find matches
        matches = self.matcher(doc)
        
        # Extract matched skills
        found_skills = set()
        for match_id, start, end in matches:
            span = doc[start:end]
            skill_text = span.text.lower()
            # Get original casing from our mapping
            if skill_text in self.skill_patterns:
                found_skills.add(self.skill_patterns[skill_text])
            else:
                found_skills.add(span.text)
        
        return list(found_skills)
    
    def _regex_fallback(self, text: str) -> List[str]:
        """Fallback regex-based extraction if spaCy unavailable."""
        text_lower = text.lower()
        found_skills = []
        
        sorted_skills = sorted(self.skill_list, key=len, reverse=True)
        
        for skill in sorted_skills:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.append(skill)
        
        return found_skills
    
    def extract_skills_with_context(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract skills with surrounding context for better classification.
        
        Args:
            text: The text to extract skills from
            
        Returns:
            List of dicts with skill and context info
        """
        if not SPACY_AVAILABLE or self.matcher is None:
            # Simple fallback
            return [{"skill": s, "context": ""} for s in self._regex_fallback(text)]
        
        doc = nlp(text)
        matches = self.matcher(doc)
        
        results = []
        seen = set()
        
        for match_id, start, end in matches:
            span = doc[start:end]
            skill_text = span.text.lower()
            
            if skill_text in seen:
                continue
            seen.add(skill_text)
            
            # Get surrounding context (sentence or nearby tokens)
            sent = span.sent if span.sent else span
            
            # Get the original skill casing
            original_skill = self.skill_patterns.get(skill_text, span.text)
            
            results.append({
                "skill": original_skill,
                "context": sent.text.strip(),
                "start": start,
                "end": end
            })
        
        return results


# Initialize global skill extractor
skill_extractor = SkillExtractor(KNOWN_SKILLS)


def extract_known_skills_from_text(text: str) -> List[str]:
    """
    Scan the entire resume text for known skills using spaCy PhraseMatcher.
    This catches skills mentioned outside the 'Skills' section.
    """
    return skill_extractor.extract_skills(text)


def extract_sections_from_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Main extraction function - reads PDF and returns structured data.
    Combines section-based extraction with spaCy-based skill matching.
    """
    text = read_pdf(pdf_path)
    raw_sections = segment_sections_by_headings(text)
    
    # Method 1: Extract skills from the Skills section
    section_skills = extract_skills_from_lines(raw_sections.get("skills", []))
    
    # Method 2: Use spaCy to scan entire text for known skills
    known_skills_found = extract_known_skills_from_text(text)
    
    # Method 3: Get skills with context for better understanding
    skills_with_context = skill_extractor.extract_skills_with_context(text)
    
    # Combine both lists and deduplicate
    all_skills_lower = set()
    combined_skills = []
    
    # Add section skills first (preserve original formatting)
    for skill in section_skills:
        if skill.lower() not in all_skills_lower:
            all_skills_lower.add(skill.lower())
            combined_skills.append(skill)
    
    # Add known skills found in text (if not already present)
    for skill in known_skills_found:
        if skill.lower() not in all_skills_lower:
            all_skills_lower.add(skill.lower())
            combined_skills.append(skill)
    
    return {
        "skills": combined_skills,
        "skills_from_section": section_skills,
        "skills_from_spacy": known_skills_found,
        "skills_with_context": skills_with_context,
        "education": "\n".join(raw_sections.get("education", [])),
        "experience": "\n".join(raw_sections.get("experience", [])),
        "file_name": os.path.basename(pdf_path),
    }


def extract_sections_from_text(text: str, file_name: str = "") -> Dict[str, Any]:
    """Extract sections from plain text (for .txt resumes).

    Returns the same structure as `extract_sections_from_pdf` so downstream
    classification code can operate the same way.
    """
    raw_sections = segment_sections_by_headings(text)

    # Method 1: Extract skills from the Skills section
    section_skills = extract_skills_from_lines(raw_sections.get("skills", []))

    # Method 2: Use spaCy to scan entire text for known skills
    known_skills_found = extract_known_skills_from_text(text)

    # Method 3: Get skills with context for better understanding
    skills_with_context = skill_extractor.extract_skills_with_context(text)

    # Combine both lists and deduplicate
    all_skills_lower = set()
    combined_skills = []

    for skill in section_skills:
        if skill.lower() not in all_skills_lower:
            all_skills_lower.add(skill.lower())
            combined_skills.append(skill)

    for skill in known_skills_found:
        if skill.lower() not in all_skills_lower:
            all_skills_lower.add(skill.lower())
            combined_skills.append(skill)

    return {
        "skills": combined_skills,
        "skills_from_section": section_skills,
        "skills_from_spacy": known_skills_found,
        "skills_with_context": skills_with_context,
        "education": "\n".join(raw_sections.get("education", [])),
        "experience": "\n".join(raw_sections.get("experience", [])),
        "file_name": file_name,
    }


# ============================================================
#  GEMINI CLASSIFICATION (adapted from Alison code)
# ============================================================

class ResumeClusterClassifier:
    """
    Classifies resumes into career clusters using Gemini API.
    Combines YOUR extraction code with THEIR API pattern.
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """
        Initialize the classifier with Gemini API.
        
        Args:
            api_key: Your Gemini API key
            model_name: Gemini model to use
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def create_classification_prompt(self, skills: List[str], education: str, experience: str) -> str:
        """
        Create the prompt that asks Gemini to classify the resume.
        This is the KEY part - combining extracted data with cluster definitions.
        """
        
        # Format cluster definitions for the prompt
        cluster_text = "\n".join([
            f"  Cluster {num}: {desc}" 
            for num, desc in CLUSTER_DESCRIPTIONS.items()
        ])
        
        # Format skills as a readable list
        skills_text = ", ".join(skills) if skills else "Not specified"
        
        prompt = f"""Analyze this resume and classify it into the most appropriate career cluster(s).

=== CLUSTER DEFINITIONS ===
{cluster_text}

=== RESUME INFORMATION ===

SKILLS:
{skills_text}

EDUCATION:
{education if education else "Not specified"}

EXPERIENCE:
{experience if experience else "Not specified"}

=== YOUR TASK ===
Based on the skills, education, and experience above, determine which cluster(s) this resume best fits.

Return ONLY a valid JSON object with this exact format (no markdown, no extra text):
{{
    "primary_cluster": <number 0-12>,
    "primary_cluster_name": "<cluster name>",
    "confidence": <0.0 to 1.0>,
    "top_three_clusters": [
        {{"cluster": <number>, "name": "<name>", "confidence": <0.0-1.0>}},
        {{"cluster": <number>, "name": "<name>", "confidence": <0.0-1.0>}},
        {{"cluster": <number>, "name": "<name>", "confidence": <0.0-1.0>}}
    ],
    "reasoning": "<explain what specific skills, education, or experience led to this classification>"
}}

Be specific in your reasoning - mention actual skills or experiences from the resume."""

        return prompt
    
    def classify_resume(self, skills: List[str], education: str, experience: str, 
                        retry_count: int = 3) -> Dict:
        """
        Classify a single resume using Gemini API.
        
        Args:
            skills: List of extracted skills
            education: Education section text
            experience: Experience section text
            retry_count: Number of retries on failure
            
        Returns:
            Dictionary with classification results
        """
        prompt = self.create_classification_prompt(skills, education, experience)
        
        for attempt in range(retry_count):
            try:
                response = self.model.generate_content(prompt)
                response_text = response.text.strip()
                
                # Remove markdown code blocks if present 
                if response_text.startswith("```"):
                    response_text = response_text.split("```")[1]
                    if response_text.startswith("json"):
                        response_text = response_text[4:]
                    response_text = response_text.strip()
                
                result = json.loads(response_text)
                result["success"] = True
                return result
                
            except Exception as e:
                print(f"  Attempt {attempt + 1} failed: {str(e)}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return {
                        "success": False,
                        "error": str(e),
                        "primary_cluster": -1,
                        "primary_cluster_name": "unknown",
                        "confidence": 0.0,
                        "top_three_clusters": [],
                        "reasoning": "Failed to classify"
                    }
    
    def classify_pdf(self, pdf_path: str) -> Dict:
        """
        Complete pipeline: Extract from PDF -> Classify with Gemini.
        
        Args:
            pdf_path: Path to the resume PDF
            
        Returns:
            Dictionary with extraction + classification results
        """
        print(f"\n[1/2] Extracting sections from: {pdf_path}")
        extracted = extract_sections_from_pdf(pdf_path)
        
        print(f"      Skills from section: {len(extracted.get('skills_from_section', []))}")
        print(f"      Skills from spaCy:   {len(extracted.get('skills_from_spacy', []))}")
        print(f"      Total unique skills: {len(extracted['skills'])}")
        print(f"      Education: {len(extracted['education'])} chars")
        print(f"      Experience: {len(extracted['experience'])} chars")
        
        print(f"[2/2] Classifying with Gemini...")
        classification = self.classify_resume(
            skills=extracted["skills"],
            education=extracted["education"],
            experience=extracted["experience"]
        )
        
        # Combine extraction and classification results
        return {
            "file_name": extracted["file_name"],
            "extracted": extracted,
            "classification": classification
        }
    
    def classify_folder(self, folder_path: str, output_json: str = "classified_resumes.json",
                        delay: float = 1.0) -> List[Dict]:
        """
        Process all PDFs in a folder.
        
        Args:
            folder_path: Path to folder containing PDFs
            output_json: Output file path
            delay: Delay between API calls (for rate limiting)
            
        Returns:
            List of classification results
        """
        results = []
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.pdf', '.txt'))]
        
        print(f"\n{'='*60}")
        print(f"Processing {len(pdf_files)} resume(s) from: {folder_path}")
        print(f"{'='*60}")
        
        for idx, filename in enumerate(pdf_files):
            print(f"\n[{idx+1}/{len(pdf_files)}] {filename}")
            
            file_path = os.path.join(folder_path, filename)
            if filename.lower().endswith('.pdf'):
                result = self.classify_pdf(file_path)
            else:
                # Handle .txt files: read text and extract sections directly
                try:
                    with open(file_path, 'r', encoding='utf-8') as fh:
                        text = fh.read()
                except Exception as e:
                    print(f"  Failed to read text file {filename}: {e}")
                    result = {
                        "file_name": filename,
                        "extracted": {},
                        "classification": {"success": False, "error": str(e)}
                    }
                    results.append(result)
                    continue

                extracted = extract_sections_from_text(text, file_name=filename)

                print(f"\n[1/2] Extracting sections from: {filename}")
                print(f"      Skills from section: {len(extracted.get('skills_from_section', []))}")
                print(f"      Skills from spaCy:   {len(extracted.get('skills_from_spacy', []))}")
                print(f"      Total unique skills: {len(extracted['skills'])}")
                print(f"      Education: {len(extracted['education'])} chars")
                print(f"      Experience: {len(extracted['experience'])} chars")

                print(f"[2/2] Classifying with Gemini...")
                classification = self.classify_resume(
                    skills=extracted['skills'],
                    education=extracted['education'],
                    experience=extracted['experience']
                )

                result = {
                    "file_name": filename,
                    "extracted": extracted,
                    "classification": classification
                }

            results.append(result)
            
            # Show result summary with top 3
            if result["classification"]["success"]:
                c = result["classification"]
                print(f"      TOP 3 MATCHES:")
                for i, cluster in enumerate(c.get("top_three_clusters", []), 1):
                    conf = cluster.get('confidence', 0) * 100
                    print(f"        {i}. Cluster {cluster.get('cluster')} - {cluster.get('name')} ({conf:.0f}%)")
            else:
                print(f"      FAILED: {result['classification'].get('error', 'Unknown error')}")
            
            # Rate limiting
            if idx < len(pdf_files) - 1:
                time.sleep(delay)
        
        # Save results
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"COMPLETE! Results saved to: {output_json}")
        print(f"{'='*60}")
        
        return results


# ============================================================
#  MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    
    # Initialize classifier
    classifier = ResumeClusterClassifier(api_key=API_KEY)
    
    # Option 1: Classify a single PDF
    # result = classifier.classify_pdf("path/to/resume.pdf")
    # print(json.dumps(result, indent=2))
    
    # Option 2: Classify all PDFs in a folder
    # Compute project root and use absolute, project-relative paths so script works
    project_root = Path(__file__).resolve().parent.parent
    RESUME_FOLDER = project_root / "models" / "file_reading_application" / "processed_data"
    output_json = project_root / "models" / "spacy_data" / "classified_resumes.json"

    results = classifier.classify_folder(
        folder_path=str(RESUME_FOLDER),
        output_json=str(output_json),
        delay=1.0  # 1 second between API calls
    )
    
    # Print summary with TOP 3 CLUSTERS
    print("\n" + "="*60)
    print("CLASSIFICATION SUMMARY - TOP 3 MATCHES")
    print("="*60)
    
    for r in results:
        if r["classification"]["success"]:
            c = r["classification"]
            print(f"\n{'-'*60}")
            print(f"FILE: {r['file_name']}")
            print(f"{'-'*60}")
            
            # Display top 3 clusters
            print("\n  TOP 3 CLUSTER MATCHES:")
            for i, cluster in enumerate(c.get("top_three_clusters", []), 1):
                confidence_pct = cluster.get('confidence', 0) * 100
                cluster_num = cluster.get('cluster', '?')
                cluster_name = cluster.get('name', 'Unknown')
                
                print(f"    #{i}: Cluster {cluster_num} - {cluster_name}")
                print(f"         Confidence: {confidence_pct:.0f}%")
            
            # Display reasoning
            print(f"\n  REASONING:")
            print(f"    {c.get('reasoning', 'No reasoning provided')}")
        else:
            print(f"\n[FAILED] {r['file_name']}: Classification failed")