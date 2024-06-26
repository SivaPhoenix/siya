let emailRegex = new RegExp("[wd]*@[w]*.[w]*");
let pwdRegex = new RegExp("^(?=.{6,20}$)(?![_.])(?!.*[_.]{4})[a-zA-Z0-9._]+$");
let usernameRegex = new RegExp("^(?=.{3,21}$)(?![_.])(?!.*[_.]{4})[a-zA-Z0-9._]+$");

export const getAge = (dateString) => {
	return Math.floor((new Date() - new Date(dateString)) / 1000 / 60 / 60 / 24 / 365);
};

export const validateEmail = (email) => {
	if (RegExp(emailRegex).test(email)) {
		return true;
	}

	return false;
};

export const validateUsername = async (username, readAllUsernamesFunc) => {
	if (usernameRegex.test(username)) {
		try {
			const all_usernames = await readAllUsernamesFunc();
			console.log(all_usernames);

			if (all_usernames.length === 0) {
				return true;
			}

			if (all_usernames.includes(username)) {
				return "Username already taken";
			}
			return true;
		} catch (e) {
			return e; //! RETURNING PROMISE FN INSTEAD OF ERROR VALUE OF PROMISE
		}
	}
	return "Invalid username.";
};

export const validatePassword = (password) => {
	if (!(password.length > 6 && password.length < 20)) {
		return "Password length should be between 8 and 20 characters.";
	}
	if (pwdRegex.test(password)) {
		return true;
	}

	return "Invalid password";
};

export const validateDate = (date) => {
	if (!date) {
		return "Please select a valid date.";
	}
	if (getAge(date) < 13 || getAge(date) > 133) {
		return "Too young/old. get out";
	}

	return true;
};

export const getDisplayNameFromUserName = (username) => {
	return username
		.replaceAll("_", " ")
		.replaceAll(".", " ")
		.split(" ")
		.filter((word) => word)
		.map((word) => (word ? word[0].toUpperCase() + word.slice(1).toLowerCase() : ""))
		.join(" ")
		.replace("/[0-9]/g", "");
};

export const internetPresent = () => {
	console.log("ping google");
	return fetch("https://www.google.com/", { mode: "no-cors" });
	
};




export const treatAsUTC = (date) => {
    var result = new Date(date);
    result.setMinutes(result.getMinutes() - result.getTimezoneOffset());
    return result;
}

export const daysBetween = (startDate, endDate) => {
    var millisecondsPerDay = 24 * 60 * 60 * 1000;
    return (treatAsUTC(endDate) - treatAsUTC(startDate)) / millisecondsPerDay;
}


export const toTitleCase = (str) => {
	if (!str.length) {
		return;
	}

	str = str.split(" ");
	return str
		.map((word, index) => {
			return (
				(word.length > 3
					? word.slice(0, 1).toUpperCase()
					: index === 0
					? word.slice(0, 1).toUpperCase()
					: word.slice(0, 1)) + word.slice(1)
			);
		})
		.join(" ");
};

export const getTimeBasedGreeting = () => {
	let hours = new Date().getHours();
	if (hours >= 16) {
		return "Good Evening";
	}
	if (hours >= 12) {
		return "Good Afternoon";
	}
	if (hours > 0) {
		return "Good Morning";
	}

	return "Greetings";
};

const errors = {
	"auth/user-disabled": "Your account has been disabled.",
	"auth/user-not-found": "No user found with the given email ID.",
	"auth/wrong-password": "The password you entered is incorrect.",
	"auth/invalid-email": "The entered email ID is invalid. Please recheck.",
	"auth/email-already-in-use": "The email you entered is already in use.",
};

export const getErrorFromCode = (errorCode) => {
	console.log(`%c Error received: ${errorCode}`, "color: #ee00ff");

	if (Object.keys(errors).includes(errorCode)) {
		return errors[errorCode];
	}

	let error = errorCode.split("/")[1].replaceAll("-", " ");
	return error[0].toUpperCase() + error.slice(1);
};

export const pink = (text, identifier, object) => {
	if (!object) {
		console.log(`%c ${text}`, "color: #ff12ee");
	} else {
		console.log(`%c Printing ${identifier || "the thing you wanted"}`, "color: #ff12ee");
		console.log(text);
	}
};

export const green = (text, identifier, object) => {
	if (!object) {
		console.log(`%c ${text}`, "color: #11ff21");
	} else {
		console.log(`%c Printing ${identifier || "the thing you wanted"}`, "color: #11ff21");
		console.log(text);
	}
};

export const red = (text, identifier, object) => {
	if (!object) {
		console.log(`%c ${text}`, "color: #f22");
	} else {
		console.log(`%c Printing ${identifier || "the thing you wanted"}`, "color: #f22");
		console.log(text);
	}
};
