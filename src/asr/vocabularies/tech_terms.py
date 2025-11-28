"""Technology vocabulary for ASR enhancement.

Includes common tech companies, products, and programming terms
that are frequently mispronounced or misspelled by ASR systems.

Total size: ~10KB bundled
"""

# Major tech companies and their products
TECH_COMPANIES = [
    # Big Tech
    "Apple", "Google", "Microsoft", "Amazon", "Meta", "Facebook", "Netflix",
    "Alphabet", "Tesla", "NVIDIA", "Intel", "AMD", "IBM", "Oracle", "Cisco",
    "Salesforce", "Adobe", "VMware", "Dell", "HP", "Hewlett-Packard",
    # Cloud providers
    "AWS", "Azure", "GCP", "DigitalOcean", "Linode", "Cloudflare", "Vercel",
    "Netlify", "Heroku", "Railway",
    # Social/Consumer
    "Twitter", "LinkedIn", "Snapchat", "TikTok", "Pinterest", "Reddit",
    "Discord", "Slack", "Zoom", "Teams", "WhatsApp", "Telegram",
    # Dev tools
    "GitHub", "GitLab", "Bitbucket", "Jira", "Confluence", "Notion",
    "Figma", "Sketch", "InVision", "Miro",
    # AI companies
    "OpenAI", "Anthropic", "DeepMind", "Hugging Face", "Stability AI",
    "Midjourney", "Cohere", "Perplexity",
    # Startups (frequently mentioned)
    "Stripe", "Square", "PayPal", "Shopify", "Twilio", "SendGrid",
    "MongoDB", "Elastic", "Databricks", "Snowflake", "Palantir",
]

# Tech products and services
TECH_PRODUCTS = [
    # Apple
    "iPhone", "iPad", "MacBook", "MacBook Air", "MacBook Pro", "iMac",
    "Mac Mini", "Mac Pro", "Apple Watch", "AirPods", "AirPods Pro",
    "HomePod", "Apple TV", "Vision Pro", "iOS", "macOS", "iPadOS",
    "watchOS", "tvOS", "Safari", "Xcode", "Swift", "SwiftUI",
    # Microsoft
    "Windows", "Office", "Excel", "Word", "PowerPoint", "Outlook",
    "Teams", "OneDrive", "SharePoint", "Visual Studio", "VS Code",
    "Azure", "Xbox", "Surface", "Copilot", "Bing", "Edge",
    # Google
    "Chrome", "Gmail", "Google Drive", "Google Docs", "Google Sheets",
    "YouTube", "Android", "Pixel", "Chromebook", "Gemini", "Bard",
    "TensorFlow", "Kubernetes", "BigQuery",
    # AI/ML
    "ChatGPT", "GPT-4", "GPT-3", "Claude", "Llama", "Mistral", "Gemini",
    "Stable Diffusion", "DALL-E", "Midjourney", "Whisper", "BERT",
    "Transformer", "PyTorch", "TensorFlow", "JAX", "MLX",
    # Databases
    "PostgreSQL", "MySQL", "SQLite", "MongoDB", "Redis", "Elasticsearch",
    "DynamoDB", "Cassandra", "Neo4j", "Supabase", "PlanetScale",
    # Frameworks
    "React", "Vue", "Angular", "Svelte", "Next.js", "Nuxt", "Remix",
    "Django", "Flask", "FastAPI", "Rails", "Laravel", "Spring",
    "Node.js", "Express", "Nest.js", "Deno", "Bun",
]

# Programming and technical terms
PROGRAMMING_TERMS = [
    # Languages
    "Python", "JavaScript", "TypeScript", "Rust", "Go", "Golang",
    "Java", "C++", "C#", "Ruby", "PHP", "Swift", "Kotlin", "Scala",
    "Haskell", "Elixir", "Clojure", "Erlang", "Julia", "R", "MATLAB",
    # Concepts
    "API", "REST", "GraphQL", "gRPC", "WebSocket", "OAuth", "JWT",
    "CRUD", "MVC", "MVP", "MVVM", "OOP", "FP", "TDD", "BDD",
    "CI/CD", "DevOps", "MLOps", "Kubernetes", "Docker", "Terraform",
    "microservices", "serverless", "monolith", "distributed",
    # Data
    "JSON", "YAML", "XML", "CSV", "Parquet", "Avro", "Protobuf",
    "SQL", "NoSQL", "ETL", "OLAP", "OLTP", "data lake", "data warehouse",
    # Infrastructure
    "Linux", "Ubuntu", "Debian", "CentOS", "RedHat", "Alpine",
    "nginx", "Apache", "Caddy", "HAProxy", "Traefik",
    "Prometheus", "Grafana", "Datadog", "New Relic", "Sentry",
    # Version control
    "Git", "GitHub", "GitLab", "Bitbucket", "merge", "rebase",
    "commit", "branch", "pull request", "PR", "code review",
]


def get_tech_companies() -> list[str]:
    """Get list of tech company names."""
    return TECH_COMPANIES.copy()


def get_tech_products() -> list[str]:
    """Get list of tech product names."""
    return TECH_PRODUCTS.copy()


def get_programming_terms() -> list[str]:
    """Get list of programming and technical terms."""
    return PROGRAMMING_TERMS.copy()
