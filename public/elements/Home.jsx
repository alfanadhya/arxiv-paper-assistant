// Home screen shown in the greeting message of a new chat (app.send_welcome) and redrawn on resume.
// Layout follows the AI-assistant home from 21st.dev, restyled to the light "Base" design (public/theme.css):
// logo, goal, three mode cards with example messages, then two short tips.
// Clicking an example puts it in Chainlit's message box (it is not sent: the user can edit it first).
import { useState } from "react";
import { MessageSquare, Search, FileText, BookOpen, SlidersHorizontal, ArrowUpRight } from "lucide-react";

const MODES = [
  {
    id: "research",
    label: "Research",
    icon: MessageSquare,
    desc: "Ask in your own words, or about the papers shown (“compare 1 and 2”).",
    examples: [
      "How are GNNs used for drug discovery?",
      "What has Iryna Gurevych published on fact checking?",
      "What are the main approaches to RAG evaluation?",
    ],
  },
  {
    id: "search",
    label: "Search",
    icon: Search,
    desc: "A plain ranked list of papers and surveys on a topic.",
    examples: [
      "search= RAG evaluation year=2020-",
      "search= generative AI for construction industry year=2020-",
      "search= graph neural networks author=Kipf; Welling",
    ],
  },
  {
    id: "title",
    label: "Title",
    icon: FileText,
    desc: "Find one paper by its exact title.",
    examples: [
      "title= Attention is all you need",
      "title= Deep residual learning for image recognition",
    ],
  },
];

// Put text in Chainlit's message box (a React-controlled textarea) and focus it.
function fillComposer(text) {
  const box = document.getElementById("chat-input");
  if (!box) return;
  const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
  setter.call(box, text);
  box.dispatchEvent(new Event("input", { bubbles: true }));
  box.focus();
  box.setSelectionRange(text.length, text.length);
}

function Logo() {
  return (
    <div className="home-logo" aria-hidden="true">
      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 200 200" width="100%" height="100%">
        <g clipPath="url(#home_clip)">
          <mask id="home_mask" style={{ maskType: "alpha" }} width="200" height="200" x="0" y="0"
                maskUnits="userSpaceOnUse">
            <path fill="#fff" fillRule="evenodd" clipRule="evenodd"
                  d="M100 150c27.614 0 50-22.386 50-50s-22.386-50-50-50-50 22.386-50 50 22.386 50 50 50zm0 50c55.228 0 100-44.772 100-100S155.228 0 100 0 0 44.772 0 100s44.772 100 100 100z" />
          </mask>
          <g mask="url(#home_mask)">
            <path fill="#fff" d="M200 0H0v200h200V0z" />
            <path fill="#0066FF" fillOpacity="0.33" d="M200 0H0v200h200V0z" />
            <g filter="url(#home_blur)" className="home-logo-spin">
              <path fill="#0066FF" d="M110 32H18v68h92V32z" />
              <path fill="#0044FF" d="M188-24H15v98h173v-98z" />
              <path fill="#0099FF" d="M175 70H5v156h170V70z" />
              <path fill="#00CCFF" d="M230 51H100v103h130V51z" />
            </g>
          </g>
        </g>
        <defs>
          <filter id="home_blur" width="385" height="410" x="-75" y="-104" colorInterpolationFilters="sRGB"
                  filterUnits="userSpaceOnUse">
            <feFlood floodOpacity="0" result="BackgroundImageFix" />
            <feBlend in="SourceGraphic" in2="BackgroundImageFix" result="shape" />
            <feGaussianBlur result="blur" stdDeviation="40" />
          </filter>
          <clipPath id="home_clip"><path fill="#fff" d="M0 0H200V200H0z" /></clipPath>
        </defs>
      </svg>
    </div>
  );
}

export default function Home() {
  const [open, setOpen] = useState(null);
  const active = MODES.find((m) => m.id === open);

  return (
    <div className="home">
      <Logo />
      <div className="home-label">arXiv Paper Assistant</div>
      <h1 className="home-title">Find computer science papers on arXiv</h1>
      <p className="home-lead">
        Search arXiv papers from 2007 to October 2026 in plain words, save the ones you need and export
        them to Zotero or Mendeley.
      </p>

      <div className="home-label home-label-left">Choose how to ask</div>
      <div className="home-modes">
        {MODES.map((m) => {
          const Icon = m.icon;
          const isOpen = open === m.id;
          return (
            <button key={m.id} type="button" className={"home-mode" + (isOpen ? " is-open" : "")}
                    aria-expanded={isOpen} onClick={() => setOpen(isOpen ? null : m.id)}>
              <Icon className="home-mode-icon" />
              <span className="home-mode-label">{m.label}</span>
              <span className="home-mode-desc">{m.desc}</span>
            </button>
          );
        })}
      </div>

      {active && (
        <div className="home-examples">
          <div className="home-examples-head">{active.label} examples · click one to use it</div>
          <ul>
            {active.examples.map((ex) => (
              <li key={ex}>
                <button type="button" onClick={() => fillComposer(ex)}>
                  <span>{ex}</span>
                  <ArrowUpRight className="home-example-arrow" />
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="home-tips">
        <div className="home-tip">
          <SlidersHorizontal className="home-tip-icon" />
          <span>
            Filters for <code>search=</code> and <code>title=</code>: <code>year=2021</code>,{" "}
            <code>year=2019-2022</code>, <code>author=Kipf; Welling</code>
          </span>
        </div>
        <div className="home-tip">
          <BookOpen className="home-tip-icon" />
          <span>
            Click <b>☐ Select</b> under a paper to save it. Open <b>Reading list</b> in the top bar to
            reorder and export (BibTeX, PDFs).
          </span>
        </div>
      </div>
    </div>
  );
}
