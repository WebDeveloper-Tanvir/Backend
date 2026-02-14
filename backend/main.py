"""
FastAPI Backend - Rule-based UI Generator
No LLM needed - uses templates and pattern matching
"""

# Install required packages
import sys
import subprocess

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

# Install dependencies
try:
    import uvicorn
except ImportError:
    install_package("uvicorn")
    import uvicorn

try:
    import nest_asyncio
except ImportError:
    install_package("nest_asyncio")
    import nest_asyncio

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import re
from typing import Dict, List, Any
from dataclasses import dataclass
import json

# Apply nest_asyncio to allow nested event loops (for Colab/Jupyter)
nest_asyncio.apply()

# --- BEGIN INLINED intent_parser.py content ---
@dataclass
class Intent:
    """Represents the parsed user intent"""
    primary_action: str  # create, add, modify, remove
    ui_type: str  # dashboard, form, table, card, etc.
    components: List[str]  # List of component names
    layout: str  # grid, flex, stack, etc.
    modifiers: Dict[str, any]  # Additional properties
    confidence: float  # 0-1 confidence score

class IntentParser:
    """Parses user input to determine what UI to build"""

    ACTIONS = {
        'create': ['create', 'make', 'build', 'generate', 'new'],
        'add': ['add', 'include', 'insert', 'put'],
        'modify': ['change', 'update', 'modify', 'edit', 'alter'],
        'remove': ['remove', 'delete', 'take out', 'get rid of'],
    }

    UI_TYPES = {
        'dashboard': ['dashboard', 'overview', 'summary', 'analytics'],
        'form': ['form', 'input form', 'registration', 'signup', 'login'],
        'table': ['table', 'data table', 'grid', 'list'],
        'card': ['card', 'info card', 'profile card'],
        'navbar': ['navbar', 'navigation', 'menu', 'header'],
        'sidebar': ['sidebar', 'side menu', 'drawer'],
        'modal': ['modal', 'dialog', 'popup', 'overlay'],
        'button': ['button', 'btn', 'action button'],
        'chart': ['chart', 'graph', 'visualization', 'plot'],
    }

    COMPONENTS = {
        'Button': ['button', 'btn', 'submit', 'action'],
        'Card': ['card', 'panel', 'box'],
        'Input': ['input', 'textbox', 'field', 'text field'],
        'Table': ['table', 'data table', 'grid'],
        'Chart': ['chart', 'graph', 'bar chart', 'line chart', 'pie chart'],
        'Navbar': ['navbar', 'navigation', 'top bar', 'header'],
        'Sidebar': ['sidebar', 'side panel', 'drawer'],
        'Modal': ['modal', 'dialog', 'popup'],
    }

    LAYOUTS = {
        'grid': ['grid', 'tiles', 'cards'],
        'flex': ['flex', 'flexible', 'row', 'column'],
        'stack': ['stack', 'vertical', 'horizontal'],
        'split': ['split', 'divided', 'sections'],
    }

    VARIANTS = {
        'primary': ['primary', 'main', 'important'],
        'secondary': ['secondary', 'alternate'],
        'outline': ['outline', 'outlined', 'border'],
        'ghost': ['ghost', 'transparent', 'minimal'],
    }

    COLORS = {
        'blue': ['blue'],
        'red': ['red', 'danger', 'error'],
        'green': ['green', 'success'],
        'yellow': ['yellow', 'warning'],
        'gray': ['gray', 'grey', 'neutral'],
    }

    def __init__(self):
        pass

    def parse(self, user_input: str) -> Intent:
        user_input = user_input.lower().strip()
        action = self._extract_action(user_input)
        ui_type = self._extract_ui_type(user_input)
        components = self._extract_components(user_input)
        layout = self._extract_layout(user_input)
        modifiers = self._extract_modifiers(user_input)
        confidence = self._calculate_confidence(action, ui_type, components)
        return Intent(
            primary_action=action,
            ui_type=ui_type,
            components=components,
            layout=layout,
            modifiers=modifiers,
            confidence=confidence
        )

    def _extract_action(self, text: str) -> str:
        for action, keywords in self.ACTIONS.items():
            if any(keyword in text for keyword in keywords):
                return action
        return 'create'

    def _extract_ui_type(self, text: str) -> str:
        for ui_type, keywords in self.UI_TYPES.items():
            if any(keyword in text for keyword in keywords):
                return ui_type
        if 'input' in text and 'button' in text:
            return 'form'
        elif 'card' in text or 'kpi' in text:
            return 'dashboard'
        return 'generic'

    def _extract_components(self, text: str) -> List[str]:
        found_components = []
        for component, keywords in self.COMPONENTS.items():
            if any(keyword in text for keyword in keywords):
                if component not in found_components:
                    found_components.append(component)
        if not found_components:
            found_components = self._infer_components_from_ui_type(
                self._extract_ui_type(text)
            )
        return found_components

    def _extract_layout(self, text: str) -> str:
        for layout, keywords in self.LAYOUTS.items():
            if any(keyword in text for keyword in keywords):
                return layout
        return 'flex'

    def _extract_modifiers(self, text: str) -> Dict[str, any]:
        modifiers = {}
        for variant, keywords in self.VARIANTS.items():
            if any(keyword in text for keyword in keywords):
                modifiers['variant'] = variant
                break
        for color, keywords in self.COLORS.items():
            if any(keyword in text for keyword in keywords):
                modifiers['color'] = color
                break
        if 'large' in text or 'big' in text:
            modifiers['size'] = 'large'
        elif 'small' in text or 'tiny' in text:
            modifiers['size'] = 'small'
        numbers = re.findall(r'\b(\d+)\b', text)
        if numbers:
            modifiers['count'] = int(numbers[0])
        return modifiers

    def _infer_components_from_ui_type(self, ui_type: str) -> List[str]:
        defaults = {
            'dashboard': ['Card', 'Chart'],
            'form': ['Input', 'Button'],
            'table': ['Table'],
            'card': ['Card'],
            'navbar': ['Navbar'],
            'sidebar': ['Sidebar'],
            'modal': ['Modal', 'Button'],
            'button': ['Button'],
            'chart': ['Chart'],
        }
        return defaults.get(ui_type, ['Card'])

    def _calculate_confidence(self, action: str, ui_type: str,
                            components: List[str]) -> float:
        score = 0.5
        if action != 'create':
            score += 0.1
        if ui_type != 'generic':
            score += 0.2
        if components:
            score += 0.2
        return min(score, 1.0)
# --- END INLINED intent_parser.py content ---

# --- BEGIN INLINED planner.py content ---
@dataclass
class ComponentPlan:
    """Plan for a single component"""
    type: str
    props: Dict[str, Any]
    children: List['ComponentPlan']
    position: Dict[str, int]  # row, col for grid layout

@dataclass
class UIPlan:
    """Complete UI generation plan"""
    layout_type: str
    container_props: Dict[str, Any]
    components: List[ComponentPlan]
    imports: List[str]
    reasoning: str

class Planner:
    """Creates structured plans from user intent"""

    TEMPLATES = {
        'dashboard': {
            'layout': 'grid',
            'container': {'className': 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 p-6'},
            'default_components': [
                {'type': 'Card', 'count': 3, 'variant': 'stats'},
                {'type': 'Chart', 'count': 2, 'variant': 'line'},
            ]
        },
        'form': {
            'layout': 'stack',
            'container': {'className': 'max-w-md mx-auto p-6 space-y-4'},
            'default_components': [
                {'type': 'Input', 'count': 3},
                {'type': 'Button', 'count': 1, 'variant': 'primary'},
            ]
        },
        'table': {
            'layout': 'flex',
            'container': {'className': 'w-full p-6'},
            'default_components': [
                {'type': 'Table', 'count': 1},
            ]
        },
        'navbar': {
            'layout': 'flex',
            'container': {'className': 'w-full'},
            'default_components': [
                {'type': 'Navbar', 'count': 1},
            ]
        },
    }

    def __init__(self):
        pass

    def create_plan(self, intent: Intent) -> UIPlan:
        template = self.TEMPLATES.get(intent.ui_type, self.TEMPLATES['form'])
        container_props = template['container'].copy()
        components = self._plan_components(intent, template)
        imports = self._determine_imports(components)
        reasoning = self._generate_reasoning(intent, components)
        return UIPlan(
            layout_type=template['layout'],
            container_props=container_props,
            components=components,
            imports=imports,
            reasoning=reasoning
        )

    def _plan_components(self, intent: Intent, template: Dict) -> List[ComponentPlan]:
        planned_components = []
        if intent.components:
            for idx, component_type in enumerate(intent.components):
                component = self._create_component_plan(
                    component_type,
                    intent.modifiers,
                    position={'row': idx // 3, 'col': idx % 3}
                )
                planned_components.append(component)
        else:
            for comp_def in template.get('default_components', []):
                count = comp_def.get('count', 1)
                for i in range(count):
                    component = self._create_component_plan(
                        comp_def['type'],
                        {'variant': comp_def.get('variant', 'default')},
                        position={'row': i // 3, 'col': i % 3}
                    )
                    planned_components.append(component)
        return planned_components

    def _create_component_plan(self, component_type: str, modifiers: Dict,
                               position: Dict) -> ComponentPlan:
        base_props = {
            'Button': {
                'variant': modifiers.get('variant', 'primary'),
                'children': 'Click me',
            },
            'Card': {
                'title': 'Card Title',
                'className': 'p-4',
            },
            'Input': {
                'label': 'Input Label',
                'placeholder': 'Enter value...',
            },
            'Table': {
                'columns': ['Name', 'Email', 'Role', 'Status'],
                'data': [
                    {'name': 'John Doe', 'email': 'john@example.com', 'role': 'Admin', 'status': 'Active'},
                    {'name': 'Jane Smith', 'email': 'jane@example.com', 'role': 'User', 'status': 'Active'},
                ],
            },
            'Chart': {
                'type': 'line',
                'data': [
                    {'name': 'Jan', 'value': 400},
                    {'name': 'Feb', 'value': 300},
                    {'name': 'Mar', 'value': 600},
                ],
            },
            'Navbar': {
                'brand': 'My App',
                'links': ['Home', 'About', 'Contact'],
            },
            'Sidebar': {
                'items': ['Dashboard', 'Profile', 'Settings'],
            },
            'Modal': {
                'title': 'Modal Title',
                'children': 'Modal content goes here',
            },
        }
        props = base_props.get(component_type, {}).copy()
        if 'variant' in modifiers:
            props['variant'] = modifiers['variant']
        if 'color' in modifiers:
            props['color'] = modifiers['color']
        if 'size' in modifiers:
            props['size'] = modifiers['size']
        return ComponentPlan(
            type=component_type,
            props=props,
            children=[],
            position=position
        )

    def _determine_imports(self, components: List[ComponentPlan]) -> List[str]:
        component_types = set(comp.type for comp in components)
        imports = [
            f"import {comp_type} from '@/components/ui/{comp_type}';"
            for comp_type in component_types
        ]
        if 'Chart' in component_types:
            imports.append("import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';")
        return imports

    def _generate_reasoning(self, intent: Intent,
                           components: List[ComponentPlan]) -> str:
        component_names = [comp.type for comp in components]
        unique_components = list(set(component_names))
        reasoning = f"Created a {intent.ui_type} layout using "
        reasoning += f"{len(components)} component(s): {', '.join(unique_components)}. "
        if intent.layout:
            reasoning += f"Arranged in a {intent.layout} layout. "
        if intent.modifiers:
            reasoning += f"Applied modifiers: {intent.modifiers}."
        return reasoning
# --- END INLINED planner.py content ---

# --- BEGIN INLINED code_generator.py content ---
class CodeGenerator:
    """Generates React code from UI plans"""

    def __init__(self):
        pass

    def generate(self, plan: UIPlan) -> str:
        imports = self._generate_imports(plan.imports)
        component_code = self._generate_component(plan)
        full_code = f"{imports}\n\n{component_code}"
        return full_code

    def _generate_imports(self, imports: List[str]) -> str:
        if not imports:
            return ""
        return "\n".join(imports)

    def _generate_component(self, plan: UIPlan) -> str:
        components_jsx = []
        for comp_plan in plan.components:
            jsx = self._generate_component_jsx(comp_plan)
            components_jsx.append(jsx)
        container_class = plan.container_props.get('className', '')
        code = f"""export default function GeneratedComponent() {{
  return (
    <div className=\"{container_class}\">
{self._indent(chr(10).join(components_jsx), 6)}
    </div>
  );
}}"""
        return code

    def _generate_component_jsx(self, comp_plan: ComponentPlan) -> str:
        comp_type = comp_plan.type
        props = comp_plan.props
        if comp_type == 'Button':
            return self._generate_button(props)
        elif comp_type == 'Card':
            return self._generate_card(props)
        elif comp_type == 'Input':
            return self._generate_input(props)
        elif comp_type == 'Table':
            return self._generate_table(props)
        elif comp_type == 'Chart':
            return self._generate_chart(props)
        elif comp_type == 'Navbar':
            return self._generate_navbar(props)
        elif comp_type == 'Sidebar':
            return self._generate_sidebar(props)
        elif comp_type == 'Modal':
            return self._generate_modal(props)
        else:
            return f"<div>Unsupported component: {comp_type}</div>"

    def _generate_button(self, props: Dict) -> str:
        variant = props.get('variant', 'primary')
        children = props.get('children', 'Button')
        return f'<Button variant=\"{variant}\">{children}</Button>'

    def _generate_card(self, props: Dict) -> str:
        title = props.get('title', 'Card Title')
        content = props.get('content', 'Card content goes here.')
        return f'''<Card>
  <Card.Title>{title}</Card.Title>
  <Card.Content>
    <p>{content}</p>
  </Card.Content>
</Card>'''

    def _generate_input(self, props: Dict) -> str:
        label = props.get('label', 'Label')
        placeholder = props.get('placeholder', 'Enter value...')
        return f'<Input label=\"{label}\" placeholder=\"{placeholder}\" />'

    def _generate_table(self, props: Dict) -> str:
        columns = props.get('columns', ['Column 1', 'Column 2'])
        data = props.get('data', [])
        col_defs = ', '.join([f'{{ header: \"{col}\", accessor: \"{col.lower().replace(" ", "_")}\" }}'
                             for col in columns])
        data_str = json.dumps(data, indent=2)
        return f'''<Table>
  columns={{[{col_defs}]}}
  data={{data}}
/>'''

    def _generate_chart(self, props: Dict) -> str:
        chart_type = props.get('type', 'line')
        data = props.get('data', [])
        data_str = json.dumps(data, indent=2)
        if chart_type == 'line':
            return f'''<Chart type=\"line\" data={{{data_str}}} />'''
        elif chart_type == 'bar':
            return f'''<Chart type=\"bar\" data={{{data_str}}} />'''
        else:
            return f'''<Chart type=\"line\" data={{{data_str}}} />'''

    def _generate_navbar(self, props: Dict) -> str:
        brand = props.get('brand', 'Brand')
        links = props.get('links', ['Home', 'About'])
        return f'''<Navbar brand=\"{brand}\">
  {' '.join([f'<Navbar.Link>{link}</Navbar.Link>' for link in links])}
</Navbar>'''

    def _generate_sidebar(self, props: Dict) -> str:
        items = props.get('items', ['Item 1', 'Item 2'])
        items_jsx = '\n  '.join([f'<Sidebar.Item>{item}</Sidebar.Item>' for item in items])
        return f'''<Sidebar>
  {items_jsx}
</Sidebar>'''

    def _generate_modal(self, props: Dict) -> str:
        title = props.get('title', 'Modal Title')
        children = props.get('children', 'Modal content')
        return f'''<Modal>
  <Modal.Title>{title}</Modal.Title>
  <Modal.Content>
    <p>{children}</p>
  </Modal.Content>
  <Modal.Footer>
    <Button variant=\"primary\">Save</Button>
    <Button variant=\"secondary\">Cancel</Button>
  </Modal.Footer>
</Modal>'''

    def _indent(self, text: str, spaces: int) -> str:
        indent = ' ' * spaces
        return '\n'.join(indent + line if line.strip() else line
                        for line in text.split('\n'))
# --- END INLINED code_generator.py content ---

# --- BEGIN INLINED code_validator.py content ---
@dataclass
class ValidationResult:
    """Result of code validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]

class CodeValidator:
    """Validates generated React code"""

    ALLOWED_COMPONENTS = [
        'Button', 'Card', 'Input', 'Table', 'Chart',
        'Navbar', 'Sidebar', 'Modal'
    ]

    REQUIRED_IMPORTS = {
        'Button': "import Button from '@/components/ui/Button';",
        'Card': "import Card from '@/components/ui/Card';",
        'Input': "import Input from '@/components/ui/Input';",
        'Table': "import Table from '@/components/ui/Table';",
        'Chart': "import Chart from '@/components/ui/Chart';",
        'Navbar': "import Navbar from '@/components/ui/Navbar';",
        'Sidebar': "import Sidebar from '@/components/ui/Sidebar';",
        'Modal': "import Modal from '@/components/ui/Modal';",
    }

    def __init__(self):
        pass

    def validate(self, code: str) -> ValidationResult:
        errors = []
        warnings = []
        suggestions = []
        syntax_errors = self._check_syntax(code)
        errors.extend(syntax_errors)
        component_errors = self._check_components(code)
        errors.extend(component_errors)
        import_warnings = self._check_imports(code)
        warnings.extend(import_warnings)
        prop_warnings = self._check_props(code)
        warnings.extend(prop_warnings)
        best_practice_suggestions = self._check_best_practices(code)
        suggestions.extend(best_practice_suggestions)
        a11y_suggestions = self._check_accessibility(code)
        suggestions.extend(a11y_suggestions)
        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )

    def _check_syntax(self, code: str) -> List[str]:
        errors = []
        open_tags = re.findall(r'<(\w+)[^/>]*>', code)
        close_tags = re.findall(r'</(\w+)>', code)
        self_closing = re.findall(r'<(\w+)[^>]*/>', code)
        for tag in self_closing:
            if tag in open_tags:
                open_tags.remove(tag)
        if len(open_tags) != len(close_tags):
            errors.append(f"Mismatched JSX tags: {len(open_tags)} open, {len(close_tags)} close")
        open_braces = code.count('{')
        close_braces = code.count('}')
        if open_braces != close_braces:
            errors.append(f"Mismatched braces: {open_braces} open, {close_braces} close")
        open_parens = code.count('(')
        close_parens = code.count(')')
        if open_parens != close_parens:
            errors.append(f"Mismatched parentheses: {open_parens} open, {close_parens} close")
        if 'export default' not in code:
            errors.append("Missing 'export default' statement")
        return errors

    def _check_components(self, code: str) -> List[str]:
        errors = []
        components_used = re.findall(r'<(\w+)', code)
        html_elements = ['div', 'span', 'p', 'h1', 'h2', 'h3', 'section', 'article']
        components_used = [c for c in components_used if c not in html_elements]
        for component in set(components_used):
            base_component = component.split('.')[0]
            if base_component not in self.ALLOWED_COMPONENTS:
                errors.append(f"Unauthorized component used: {component}")
        return errors

    def _check_imports(self, code: str) -> List[str]:
        warnings = []
        components_used = re.findall(r'<(\w+)', code)
        components_used = [c.split('.')[0] for c in components_used]
        for component in set(components_used):
            if component in self.ALLOWED_COMPONENTS:
                required_import = self.REQUIRED_IMPORTS[component]
                if required_import not in code:
                    warnings.append(f"Missing import for component: {component}")
        return warnings

    def _check_props(self, code: str) -> List[str]:
        warnings = []
        if 'style={{' in code or 'style={' in code:
            warnings.append("Inline styles detected - use Tailwind classes instead")
        if 'className=' not in code:
            warnings.append("No className detected - consider adding Tailwind classes")
        return warnings

    def _check_best_practices(self, code: str) -> List[str]:
        suggestions = []
        if '.map(' in code and 'key=' not in code:
            suggestions.append("Consider adding 'key' prop when rendering lists")
        if not re.search(r'export default function [A-Z]\w+', code):
            suggestions.append("Component name should be PascalCase")
        if 'function' in code and 'props.' in code:
            suggestions.append("Consider destructuring props in function signature")
        return suggestions

    def _check_accessibility(self, code: str) -> List[str]:
        suggestions = []
        buttons = re.findall(r'<Button[^>]*>', code)
        for button in buttons:
            if 'aria-label' not in button and '>' not in button:
                suggestions.append("Add aria-label or text content to buttons")
        if '<img' in code and 'alt=' not in code:
            suggestions.append("Add alt text to images for accessibility")
        return suggestions

    def fix_common_issues(self, code: str) -> str:
        fixed_code = code
        fixed_code = re.sub(r'(import .+)(?<!;)$', r'\1;', fixed_code, flags=re.MULTILINE)
        fixed_code = re.sub(r'([>}])\s*([<{])', r'\1\n\2', fixed_code)
        return fixed_code
# --- END INLINED code_validator.py content ---


# Initialize FastAPI app
app = FastAPI(
    title="Rule-Based UI Generator",
    description="Generate React UIs without LLM - pure template matching",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add logging middleware to debug requests
@app.middleware("http")
async def log_requests(request, call_next):
    print(f"📥 Incoming request: {request.method} {request.url.path}")
    print(f"   Headers: {dict(request.headers)}")
    response = await call_next(request)
    print(f"📤 Response status: {response.status_code}")
    return response

# Initialize pipeline components
intent_parser = IntentParser()
planner = Planner()
code_generator = CodeGenerator()
code_validator = CodeValidator()

# Request/Response models
class GenerateRequest(BaseModel):
    """Request model for UI generation"""
    prompt: str
    current_code: Optional[str] = None

class GenerateResponse(BaseModel):
    """Response model for UI generation"""
    code: str
    explanation: str
    plan: dict
    validation: dict

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Rule-Based UI Generator API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "generate": "/api/generate",
            "generate_v1": "/api/v1/generate",
            "templates": "/api/v1/templates",
            "examples": "/api/v1/examples",
            "docs": "/docs"
        },
        "note": "Both /api/generate and /api/v1/generate endpoints are available"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pipeline": "rule-based",
        "components": {
            "intent_parser": "active",
            "planner": "active",
            "code_generator": "active",
            "code_validator": "active"
        }
    }

async def _generate_ui_logic(request: GenerateRequest) -> GenerateResponse:
    """
    Core UI generation logic (shared between endpoints)
    """
    prompt = request.prompt.strip()

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    # Step 1: Parse intent
    print(f"📝 Parsing intent: {prompt[:50]}...")
    intent = intent_parser.parse(prompt)
    print(f"✅ Intent parsed: {intent.ui_type} with {len(intent.components)} components")

    # Step 2: Create plan
    print(f"📋 Creating plan...")
    plan = planner.create_plan(intent)
    print(f"✅ Plan created: {len(plan.components)} components planned")

    # Step 3: Generate code
    print(f"🔨 Generating code...")
    code = code_generator.generate(plan)
    print(f"✅ Code generated: {len(code)} characters")

    # Step 4: Validate code
    print(f"✔️  Validating code...")
    validation_result = code_validator.validate(code)
    print(f"✅ Validation complete: {'PASS' if validation_result.is_valid else 'FAIL'}")

    # If validation fails, try to fix
    if not validation_result.is_valid:
        print(f"🔧 Attempting to fix issues...")
        code = code_validator.fix_common_issues(code)
        validation_result = code_validator.validate(code)

    # Prepare response
    response = GenerateResponse(
        code=code,
        explanation=plan.reasoning,
        plan={
            "layout": plan.layout_type,
            "components": [
                {"type": c.type, "props": c.props}
                for c in plan.components
            ],
            "imports": plan.imports
        },
        validation={
            "is_valid": validation_result.is_valid,
            "errors": validation_result.errors,
            "warnings": validation_result.warnings,
            "suggestions": validation_result.suggestions
        }
    )

    print(f"🎉 Generation complete!")
    return response

@app.post("/api/v1/generate", response_model=GenerateResponse)
async def generate_ui_v1(request: GenerateRequest):
    """
    Generate UI from natural language prompt (v1 endpoint)

    Pipeline:
    1. Parse intent from prompt
    2. Create structured plan
    3. Generate React code
    4. Validate code
    5. Return result
    """
    try:
        return await _generate_ui_logic(request)
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate", response_model=GenerateResponse)
async def generate_ui(request: GenerateRequest):
    """
    Generate UI from natural language prompt (legacy endpoint for frontend compatibility)
    """
    try:
        return await _generate_ui_logic(request)
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/templates")
async def list_templates():
    """List available UI templates"""
    return {
        "templates": list(planner.TEMPLATES.keys()),
        "components": list(intent_parser.COMPONENTS.keys())
    }

@app.get("/api/v1/examples")
async def get_examples():
    """Get example prompts"""
    return {
        "examples": [
            {
                "prompt": "Create a dashboard with 3 cards and 2 charts",
                "type": "dashboard"
            },
            {
                "prompt": "Build a login form with email and password inputs",
                "type": "form"
            },
            {
                "prompt": "Make a data table with 5 columns",
                "type": "table"
            },
            {
                "prompt": "Create a navbar with logo and menu items",
                "type": "navbar"
            },
            {
                "prompt": "Add a modal with title and buttons",
                "type": "modal"
            }
        ]
    }

# Run the application
def run_server():
    """Run the FastAPI server"""
    print("🚀 Starting Rule-Based UI Generator...")
    print("📚 No LLM needed - using pure template matching!")
    print("🌐 API will be available at http://0.0.0.0:8000")
    print("📖 Docs available at http://0.0.0.0:8000/docs")
    
    # Use config instead of run() for better compatibility with Jupyter
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
    server = uvicorn.Server(config)
    
    # Run in the current event loop
    import asyncio
    loop = asyncio.get_event_loop()
    loop.create_task(server.serve())
    
    print("✅ Server started! Press Ctrl+C to stop.")
    return server

if __name__ == "__main__":
    server = run_server()