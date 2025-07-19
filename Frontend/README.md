# ArogyaAI Frontend

ArogyaAI Frontend is a modern, responsive React application for healthcare providers to manage vaccine distribution, predict demand, identify dropout risks, and detect zero-dose clusters using AI-powered analytics.

## Features
- **Vaccine Demand Forecasting:** Predict short-term vaccine demand per district using LSTM models and climate data.
- **Dropout Risk Prediction:** Identify children at risk of missing vaccinations for early intervention.
- **Zero-Dose Cluster Detection:** Detect and prioritize high-risk areas for targeted vaccination efforts.
- **Beautiful, Responsive UI:** Built with Tailwind CSS and React for a seamless experience on all devices.

## Tech Stack
- [React](https://reactjs.org/)
- [Vite](https://vitejs.dev/)
- [Tailwind CSS](https://tailwindcss.com/)
- [Heroicons](https://heroicons.com/) (for icons)

## Getting Started

### Prerequisites
- Node.js (v16 or above recommended)
- npm or yarn

### Installation
1. **Install dependencies:**
   ```bash
   npm install
   # or
   yarn install
   ```
2. **Start the development server:**
   ```bash
   npm run dev
   # or
   yarn dev
   ```
3. Open [http://localhost:5173](http://localhost:5173) in your browser.

### Build for Production
```bash
npm run build
# or
yarn build
```
The production-ready files will be in the `dist/` folder.

### Linting & Formatting
```bash
npm run lint
# or
yarn lint
```

## Project Structure
```
Frontend/
  src/
    components/    # Reusable UI components
    pages/         # Main app pages (Home, Features, Forecasting, Dropout, Clustering, etc.)
    assets/        # Images, icons, etc.
  public/          # Static files
  tailwind.config.js
  vite.config.js
  ...
```

## API Integration
- The frontend expects the backend API to be running (see backend README for details).
- Update API URLs in the code if your backend runs on a different host/port.

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License.

## Contact
For questions or support, please contact the project maintainer.
